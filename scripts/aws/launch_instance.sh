#!/usr/bin/env bash
# AWS instance lifecycle for TRACEGUARD v0.7 (docs/AWS_SETUP.md).
#
# Launch a p4de.24xlarge (8x A100-80GB) with the Deep Learning AMI (Ubuntu),
# an EBS gp3 volume for model weights, and an EFA-enabled placement group.
# Everything is explicit — no hidden defaults; check prices before running
# (p4de.24xlarge: $27.447/hr on-demand us-east-1, spot from ~$13-22/hr,
#  verified 2026-11-06 via instances.vantage.sh / aws.amazon.com pricing).
#
# Usage: ./launch_instance.sh <instance-type> <key-name> [spot]
set -euo pipefail

INSTANCE_TYPE="${1:-p4de.24xlarge}"
KEY_NAME="${2:?usage: launch_instance.sh <instance-type> <key-name> [spot]}"
SPOT="${3:-}"
REGION="${AWS_REGION:-us-east-1}"

# EBS: 2.5 TB gp3 covers Qwen3-14B + 32B-FP8 + gpt-oss + KV scratch with headroom.
cat > /tmp/tg-user-data.sh <<'EOF'
#!/bin/bash
set -euo pipefail
mkfs -t xfs /dev/nvme2n1 || true
mkdir -p /mnt/ebs
echo "$(lsblk -no UUID /dev/nvme2n1 2>/dev/null || blkid -s UUID -o value /dev/nvme2n1) /mnt/ebs xfs defaults,nofail 0 2" >> /etc/fstab
mount -a
mkdir -p /mnt/ebs/hf-cache /mnt/ebs/runs
chown -R ubuntu:ubuntu /mnt/ebs
# NVIDIA driver + DCGM health probe: fail fast on degraded GPUs.
nvidia-smi || (echo 'NVIDIA driver missing'; exit 1)
nvidia-smi --query-gpu=name,memory.total,pci.device_id --format=csv
EOF

AMI=$(aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning AMI GPU PyTorch *Ubuntu*" \
            "Name=state,Values=available" \
  --query "sort_by(Images,&CreationDate)[-1].ImageId" \
  --output text --region "$REGION")
echo "AMI: $AMI"

# Placement group: EFA/100Gbps fabric wants cluster placement.
PG_NAME="traceguard-pg"
aws ec2 create-placement-group --group-name "$PG_NAME" --strategy cluster \
  --region "$REGION" 2>/dev/null || true

SECURITY_GROUP=$(aws ec2 create-security-group \
  --group-name traceguard-ssh-only --description "TRACEGUARD: SSH only" \
  --output text --region "$REGION" 2>/dev/null || \
  aws ec2 describe-security-groups --group-names traceguard-ssh-only \
    --query "SecurityGroups[0].GroupId" --output text --region "$REGION")
aws ec2 authorize-security-group-ingress --group-id "$SECURITY_GROUP" \
  --protocol tcp --port 22 --cidr 0.0.0.0/0 --region "$REGION" 2>/dev/null || true

INSTANCE_PARAMS=(
  --image-id "$AMI"
  --instance-type "$INSTANCE_TYPE"
  --key-name "$KEY_NAME"
  --security-group-ids "$SECURITY_GROUP"
  --placement "GroupName=${PG_NAME}"
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3","DeleteOnTermination":true}},{"DeviceName":"/dev/nvme2n1","VirtualName":"ephemeral0"}]'
  --user-data file:///tmp/tg-user-data.sh
  --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=traceguard},{Key=Billing,Value=aws-research-credits}]"
  --region "$REGION"
)

if [[ "$SPOT" == "spot" ]]; then
  # Spot: up to ~60-70% off on-demand; interruptions are EXPECTED —
  # the workflow journal + run transactions make shards resumable.
  INSTANCE_PARAMS+=(--instance-market-options '{"MarketType":"spot"}')
fi

INSTANCE_ID=$(aws ec2 run-instances "${INSTANCE_PARAMS[@]}" \
  --query "Instances[0].InstanceId" --output text)
echo "launched: $INSTANCE_ID ($INSTANCE_TYPE spot=${SPOT:-no})"
aws ec2 wait instance-status-ok --instance-ids "$INSTANCE_ID" --region "$REGION"
DNS=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query "Reservations[0].Instances[0].PublicDnsName" --output text --region "$REGION")
echo "ssh -A ubuntu@$DNS"
echo "remember: terminate with 'aws ec2 terminate-instances --instance-ids $INSTANCE_ID'"
