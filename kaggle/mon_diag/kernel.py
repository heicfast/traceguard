# TRACEGUARD monitor-attention diagnostics (NOT part of the research pipeline).
# Decisive experiment for the 2026-10-20 monitor-kernel OOM forensics:
# 1. does attn_implementation="sdpa" actually take effect on T4 + fp16?
# 2. which attention path runs and what peak VRAM do increasing prompt
#    lengths actually consume with generate()?
# Writes /kaggle/working/mon_diag.json; failed cases are DATA (exit 0 anyway).

import json
import platform
import time
from datetime import UTC, datetime

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B"
OUT = "/kaggle/working/mon_diag.json"
report = {
    "kernel": "traceguard-mon-diag",
    "ts_utc": datetime.now(UTC).isoformat(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "cases": [],
}


def case(name, fn):
    entry = {"case": name}
    t0 = time.time()
    try:
        fn(entry)
    except Exception as exc:  # noqa: BLE001 - failures are data
        entry["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    entry["seconds"] = round(time.time() - t0, 1)
    report["cases"].append(entry)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"CASE {name}: {json.dumps(entry, default=str)[:600]}", flush=True)


def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
    model.eval()

    def attn_state(entry):
        entry["config_attn_implementation"] = getattr(model.config, "_attn_implementation", None)
        layer0 = model.model.layers[0].self_attn
        entry["attn_class"] = type(layer0).__name__
        entry["hf_device_map"] = str(model.hf_device_map)[:200]

    case("attn_state", attn_state)

    def probe(prompt_tokens: int, entry):
        filler = "The maintenance log entry describes routine rotation. "
        text = "Routine software maintenance audit transcript.\n" + filler * (prompt_tokens // 12 + 1)
        ids = tok([text], return_tensors="pt")
        n = ids["input_ids"].shape[-1]
        if n > prompt_tokens:
            ids = {k: v[:, :prompt_tokens] for k, v in ids.items()}
        n = ids["input_ids"].shape[-1]
        ids = ids.to(model.device)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**ids, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        entry["prompt_tokens"] = n
        entry["generated_ok"] = True
        entry["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        entry["seconds"] = round(time.time() - t0, 1)
        entry["new_tokens"] = int(out.shape[-1] - n)

    for n in (8000, 12000, 16000, 20000, 24000, 30000):
        case(f"generate_{n}tok", lambda e, n=n: probe(n, e))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)
    print("MON DIAG COMPLETE", json.dumps(report["cases"], default=str)[:1500], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
