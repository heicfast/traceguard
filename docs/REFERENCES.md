# TRACEGUARD reference library

Curated bibliography grounding the v0.7 program. Every entry carries a
**verification key**:

- **[V]** — link opened and confirmed during the 2026-11-06 upgrade research pass;
- **[R]** — carried from this repository's frozen protocol docs (already vetted
  by the project's review process);
- **[C]** — canonical, stable citation (pre-2025 classic; arXiv id/venue from
  the original record).

Entries without a stable identifier cite author–title–venue–year only; we
never guess an arXiv id. Additions after any preregistration freeze must be
append-only.

---

## 1. Chain-of-thought monitorability and AI control

1. **[V]** Korbak, T., Balesni, M., Barnes, E., et al. (46 authors incl. UK AISI, Apollo Research, METR, Anthropic, OpenAI, Google DeepMind). *Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety.* arXiv:2507.11473, 2025. https://arxiv.org/abs/2507.11473
2. **[V]** Guan, M. Y., Wang, M., Carroll, M., Dou, Z., Wei, A. Y., Williams, M., Arnav, B., Huizinga, J., Kivlichan, I., Glaese, M., Pachocki, J., Baker, B. *Monitoring Monitorability.* arXiv:2512.18311, 2025. https://arxiv.org/abs/2512.18311
3. **[V]** OpenAI. *Evaluating chain-of-thought monitorability.* Dec 18, 2025. https://openai.com/index/evaluating-chain-of-thought-monitorability
4. **[V]** OpenAI. *Open Sourcing Monitorability Evaluations.* Apr 23, 2026. https://openai.com/index/article/open-sourcing-monitorability-evaluations
5. **[V]** Greenblatt, R., Shlegeris, B., Sachan, K., Roger, F. *AI Control: Improving Safety Despite Intentional Subversion.* ICML 2024. arXiv:2312.06942. https://arxiv.org/abs/2312.06942
6. **[V]** Kutasov, J., et al. (METR). *SHADE-Arena: Evaluating Sabotage and Monitoring in LLM Agents.* arXiv:2506.15740, 2025. https://arxiv.org/abs/2506.15740
7. **[V]** Najt, E., Toft, C., Tracy, T., Roger, F., Benton, J. *SLEIGHT-Bench: A Benchmark of Evasion Attacks Against Agent Monitors.* Anthropic, 2026. https://alignment.anthropic.com/2026/sleight-bench/
8. **[R]** Kutasov, J., et al. *Evaluating Control Protocols for Untrusted AI Agents.* AAAI 2026 workshop.
9. **[R]** Wang, et al. *MonitorBench.* arXiv:2603.28590, 2026.
10. **[R]** Jotautaitė, et al. *MonitoringBench.* arXiv:2605.09684, 2026.
11. **[R]** OpenAI. *Reasoning Models Struggle to Control their Chains of Thought.* 2026. https://openai.com/index/reasoning-models-chain-of-thought-controllability/
12. **[R]** Shiromani, & Richter. *A False Average.* arXiv:2608.00583, 2026 (preprint; requires independent replication).
13. **[V]** UK AI Security Institute. *Inspect AI.* https://inspect.aisi.org.uk/
14. **[V]** UK AI Security Institute. *ControlArena.* https://control-arena.aisi.org.uk/

## 2. Chain-of-thought faithfulness and reasoning transparency

15. **[C]** Turpin, M., Michael, J., Perez, E., Bowman, S. R. *Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting.* NeurIPS 2023. arXiv:2305.04388.
16. **[C]** Lanham, T., et al. *Measuring Faithfulness in Chain-of-Thought Reasoning.* arXiv:2307.13702, 2023 (Anthropic).
17. **[R]** Chen et al. *Reasoning Models Don't Always Say What They Think.* arXiv:2505.05410, 2025 (Anthropic). https://arxiv.org/abs/2505.05410
18. **[C]** Pfau, J., Letarte, A., Goel, G., et al. *Let's Think Dot by Dot: Hidden Computation in Chain-of-Thought Prompting.* ACL 2024. arXiv:2403.01644.
19. **[C]** Jain, N., et al. *Mechanistically Analyzing the Effects of Fine-Tuning on Unfaithful Chain-of-Thought Reasoning.* 2024.
20. **[C]** Sharma, M., et al. *Towards Understanding Sycophancy in Language Models.* ICLR 2024. arXiv:2310.13548.

## 3. Agent safety, scheming, sabotage, and control evaluations

21. **[V]** Meinke, A., Schoen, B., Scheurer, J., Balesni, M., Shah, R., Hobbhahn, M. (Apollo Research). *Frontier Models are Capable of In-context Scheming.* arXiv:2412.04984, 2024. https://arxiv.org/abs/2412.04984
22. **[C]** Denison, C., et al. (Anthropic). *Sycophancy to Subterfuge: Investigating Reward Tampering in Language Models.* 2024. arXiv:2406.10162.
23. **[C]** Greenblatt, R., et al. (Anthropic). *Alignment Faking in Large Language Models.* arXiv:2412.14093, 2024.
24. **[C]** Hubinger, E., et al. *Model Organisms of Misalignment: The Case for a New Science of Evaluating and Monitoring Dangerous Capabilities.* arXiv:2404.15782, 2024.
25. **[C]** Greenblatt, R., et al. (Anthropic). *Language Models Can Strategically Deceive Their Users When Put Under Pressure.* Alignment Forum, 2024.
26. **[C]** Debenedetti, E., et al. *AgentDojo: A Dynamic Environment to Evaluate Attacks and Defenses for LLM Agents.* arXiv:2406.13352, 2024.
27. **[C]** Andriushchenko, M., et al. *AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents.* arXiv:2410.01624, 2024.
28. **[C]** Pacchiardi, L., et al. *How to Catch an AI Liar: Lie Detection in Black-Box LLMs by Asking Unrelated Questions.* ICLR 2024.
29. **[C]** Greshake, K., et al. *Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.* ACM AISec 2023. arXiv:2302.12173.
30. **[C]** Zou, A., et al. *Universal and Transferable Adversarial Attacks on Aligned Text Generators.* ICML 2024. arXiv:2307.15043.
31. **[C]** Wei, A., et al. *Jailbroken: How Does LLM Safety Training Fail?* NeurIPS 2023.
32. **[C]** Anthropic. *Many-shot Jailbreaking.* 2024. https://www.anthropic.com/research/many-shot-jailbreaking
33. **[C]** Perez, E., et al. *Red Teaming Language Models with Language Models.* arXiv:2209.07858, 2022.
34. **[R]** Zhang et al. *Agent Safety Bench (ASB).* 2024.

## 4. LLM evaluation and judge reliability

35. **[C]** Chang, Y., et al. *A Survey on Evaluation of Large Language Models.* ACM TIST 2024. arXiv:2305.18290.
36. **[C]** Zheng, L., et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023 Datasets & Benchmarks. arXiv:2306.05685.
37. **[C]** Wang, P., et al. *Large Language Models are not Fair Evaluators.* ACL 2024. arXiv:2305.17926.
38. **[C]** Boubdir, M., et al. *Eliciting Transparent Probabilities from LLMs.* 2023.
39. **[C]** Chiang, W., et al. *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference.* 2024. arXiv:2403.04132.
40. **[C]** Cohen, J. *A Coefficient of Agreement for Nominal Scales.* Educational and Psychological Measurement 20(1), 1960.

## 5. Statistical methodology

### Bootstrap, clustering, paired designs

41. **[C]** Efron, B. *Bootstrap Methods: Another Look at the Jackknife.* The Annals of Statistics 7(1), 1979.
42. **[C]** Efron, B. *Better Bootstrap Confidence Intervals.* JASA 82(397), 1987.
43. **[C]** Field, C. A., Welsh, A. H. *Bootstrapping Clustered Data.* JRSS-B 69(3), 2007.
44. **[C]** Cameron, A. C., Miller, D. L. *A Practitioner's Guide to Cluster-Robust Inference.* Journal of Human Resources 50(2), 2015.

### ROC / PR methodology

45. **[C]** DeLong, E. R., DeLong, D. M., Clarke-Pearson, D. L. *Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach.* Biometrics 44(3), 1988.
46. **[C]** Hanley, J. A., McNeil, B. J. *The Meaning and Use of the Area under a Receiver Operating Characteristic (ROC) Curve.* Radiology 143(1), 1982.
47. **[C]** Saito, T., Rehmsmeier, M. *The Precision-Recall Plot is More Informative than the ROC Plot when Evaluating Binary Classifiers on Imbalanced Datasets.* PLoS ONE 10(3), 2015.
48. **[C]** Davis, J., Goadrich, M. *The Relationship Between Precision-Recall and ROC Curves.* ICML 2006.
49. **[C]** Fawcett, T. *An Introduction to ROC Analysis.* Pattern Recognition Letters 27(8), 2006.

### Multiple comparisons and selective inference

50. **[C]** Holm, S. *A Simple Sequentially Rejective Multiple Test Procedure.* Scandinavian Journal of Statistics 6(2), 1979.
51. **[C]** Benjamini, Y., Hochberg, Y. *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing.* JRSS-B 57(1), 1995.
52. **[C]** Benjamini, Y., Yekutieli, D. *The Control of the False Discovery Rate in Multiple Testing under Dependency.* Annals of Statistics 29(4), 2001.
53. **[C]** Gelman, A., Loken, E. *The Statistical Crisis in Science: Data-dependent Analysis — a "Garden of Forking Paths" — Explains Why Many Statistically Significant Comparisons Don't Hold Up.* American Scientist 102(6), 2014.
54. **[C]** Kerr, N. L. *HARKing: Hypothesizing After the Results are Known.* Personality and Social Psychology Review 2(3), 1998.

### Equivalence testing and effect sizes

55. **[C]** Schuirmann, D. J. *A Comparison of the Two One-Sided Tests Procedure and the Power Approach for Assessing the Equivalence of Average Bioavailability.* Journal of Pharmacokinetics and Biopharmaceutics 15(6), 1987.
56. **[C]** Lakens, D. *Equivalence Tests: A Practical Primer for t-tests, Correlations and Meta-analyses.* Social Psychological and Personality Science 8(4), 2017.

### Calibration and decision utility

57. **[C]** Brier, G. W. *Verification of Forecasts Expressed in Terms of Probability.* Monthly Weather Review 78(1), 1950.
58. **[C]** Vickers, A. J., Elkin, E. B. *Decision Curve Analysis: A Novel Method for Evaluating Prediction Models.* Medical Decision Making 26(6), 2006.
59. **[C]** Vickers, A. J., Cronin, A. M. *Traditional Statistical Methods for Evaluating Prediction Models are Uninformative.* Medical Decision Making 28(4), 2008 (letter exchange).

### Exact and small-sample inference

60. **[C]** McNemar, Q. *Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages.* Psychometrika 12(2), 1947.
61. **[C]** Clopper, C. J., Pearson, E. S. *The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial.* Biometrika 26(4), 1934.
62. **[C]** Wilson, E. B. *Probable Inference, the Law of Succession, and Statistical Inference.* JASA 22(158), 1927.
63. **[C]** Hanley, J. A., Lippman-Hand, A. *If Nothing Goes Wrong, Is Everything All Right? Interpreting Zero Numerators.* JAMA 249(13), 1983.

### Preregistration, reproducibility, reporting standards

64. **[C]** Nosek, B. A., Ebersole, C. R., DeHaven, A. C., Mellor, D. T. *The Preregistration Revolution.* PNAS 115(11), 2018.
65. **[C]** Henderson, P., et al. *Deep Reinforcement Learning that Matters.* AAAI 2018. arXiv:1709.06560.
66. **[C]** Bouthillier, X., et al. *Accounting for Variance in Machine Learning Benchmarks.* MLSys 2021.
67. **[C]** Pineau, J., et al. *Improving Reproducibility in Machine Learning Research: A Report from the NeurIPS 2019 Reproducibility Program.* arXiv:2003.12206, 2020.
68. **[V]** NeurIPS. *Paper Checklist Guidelines.* https://neurips.cc/public/guides/PaperChecklist
69. **[C]** Collins, G. S., Reitsma, J. B., Altman, D. G., Moons, K. G. M. *Transparent Reporting of a Multivariable Prediction Model for Individual Prognosis or Diagnosis (TRIPOD).* BMJ 350, 2015.
70. **[C]** Collins, G. S., et al. *TRIPOD+AI Statement: Updated Guidance for Reporting Clinical Prediction Models that Use Regression or Machine Learning Methods.* BMJ 385, 2024.
71. **[C]** von Elm, E., et al. *The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) Statement.* Lancet 370, 2007.
72. **[C]** Schulz, K. F., Altman, D. G., Moher, D. *CONSORT 2010 Statement.* BMJ 340, 2010.
73. **[V]** NIST. *AI Risk Management Framework 1.0.* NIST AI 100-1, 2023. https://www.nist.gov/itl/ai-risk-management-framework

## 6. Inference engines, kernels, and serving systems

74. **[V]** Kwon, W., Li, Z., Zhuang, S., et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023. arXiv:2309.06180. (vLLM)
75. **[V]** vLLM project. *vLLM releases (v0.33.x line, Oct 2026).* https://github.com/vllm-project/vllm/releases
76. **[V]** vLLM project. *vLLM Now Supports gpt-oss.* Aug 5, 2025. https://vllm.ai/blog/2025-08-05-gpt-oss
77. **[C]** Zheng, L., et al. *SGLang: Efficient Execution of Structured Language Model Programs.* 2024. arXiv:2312.07104 (RadixAttention).
78. **[V]** SGLang project. *SGLang releases (v0.5.22, Oct 2026).* https://github.com/sgl-project/sglang/releases
79. **[C]** Dao, T., Fu, D., Ermon, S., Rudra, A., Ré, C. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022. arXiv:2205.14135.
80. **[C]** Dao, T. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* ICLR 2024. arXiv:2307.08691.
81. **[V]** Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., Dao, T. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.* arXiv:2407.08608, 2024. https://arxiv.org/abs/2407.08608
82. **[C]** FlashInfer project. *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference.* https://github.com/flashinfer-ai/flashinfer
83. **[C]** Cai, T., et al. *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* ICML 2024. arXiv:2401.10774.
84. **[C]** Li, Y., et al. *EAGLE: Speculative Sampling Requires Revisiting Feature Uncertainty.* ICML 2024. arXiv:2401.15077.
85. **[C]** Li, Y., et al. *EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees.* ACL 2024. arXiv:2406.16858.
86. **[C]** Li, Y., et al. *EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training Time Test.* arXiv:2503.01840, 2025.
87. **[C]** Spector, B., Re, C. *Accelerating Large Language Model Decoding with Speculative Sampling.* 2023.
88. **[C]** Frantar, E., et al. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023. arXiv:2210.17323.
89. **[C]** Lin, J., et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys 2024. arXiv:2306.00978.
90. **[C]** Xiao, G., et al. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* ICML 2023. arXiv:2211.10438.
91. **[C]** Peng, B., et al. *YaRN: Efficient Context Window Extension of Large Language Models.* ICLR 2024. arXiv:2309.00071.
92. **[V]** DeepSeek-AI. *FlashMLA: efficient multi-head latent attention kernels.* https://github.com/deepseek-ai/FlashMLA
93. **[V]** DeepSeek-AI. *DeepGEMM: clean and efficient tensor core kernel library on GPU.* https://github.com/deepseek-ai/DeepGEMM
94. **[V]** DeepSeek-AI. *3FS: a high-performance distributed file system.* https://github.com/deepseek-ai/3FS
95. **[V]** DeepSeek-AI. *DualPipe: a bidirectional pipeline parallelism algorithm.* https://github.com/deepseek-ai/DualPipe
96. **[C]** Micikevicius, P., et al. *FP8 Formats for Deep Learning.* arXiv:2209.05433, 2022.

## 7. Model reports (open weights and frontier)

97. **[C]** DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* (Multi-head Latent Attention). arXiv:2405.04434, 2024.
98. **[C]** DeepSeek-AI. *DeepSeek-V3 Technical Report.* arXiv:2412.19437, 2024 (MLA, MTP, FP8 training, DualPipe).
99. **[C]** DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* arXiv:2501.12948, 2025 (Nature 2025 version).
100. **[C]** Qwen Team. *Qwen3 Technical Report.* arXiv:2505.09388, 2025. https://arxiv.org/abs/2505.09388
101. **[V]** Qwen Team. *Qwen3.5: Towards Native Multimodal Agents.* Feb 15, 2026. https://qwen.ai/blog?id=qwen3.5
102. **[V]** Qwen Team. *Qwen3.8-Max: A New Bar for Coding and Cowork.* Aug 2, 2026. https://qwen.ai/blog?id=qwen3.8-max
103. **[C]** Zeng, A., et al. *GLM-130B: An Open Bilingual Pre-trained Model.* ICLR 2023. arXiv:2210.02414.
104. **[V]** Z.ai. *GLM-4.6: Advanced Agentic, Reasoning and Coding.* Sep 30, 2025. https://z.ai/blog/glm-4.6
105. **[C]** Moonshot AI. *Kimi K2: Open Agentic Intelligence.* arXiv:2507.20534, 2025.
106. **[V]** Moonshot AI. *Kimi K3 / K3.1 Pro releases.* Jul 16 / Oct 24, 2026. https://www.moonshot.ai
107. **[C]** Dubey, A., et al. *The Llama 3 Herd of Models.* arXiv:2407.21783, 2024.
108. **[C]** Jiang, A. Q., et al. *Mistral 7B.* arXiv:2310.06825, 2023.
109. **[C]** Gemma Team. *Gemma 3 Technical Report.* arXiv:2503.19786, 2025.
110. **[V]** OpenAI. *gpt-oss-20b* (21B-total/3.6B-active MoE, native MXFP4). https://huggingface.co/openai/gpt-oss-20b
111. **[C]** OpenAI. *GPT-4 Technical Report.* arXiv:2303.08774, 2023.
112. **[C]** Ouyang, L., et al. *Training Language Models to Follow Instructions with Human Feedback (InstructGPT).* NeurIPS 2022. arXiv:2203.02155.
113. **[C]** Bai, Y., et al. *Constitutional AI: Harmlessness from AI Feedback.* arXiv:2212.08073, 2022 (Anthropic).
114. **[V]** Anthropic. *Introducing Claude Opus 4.6.* 2026. https://www.anthropic.com/news/claude-opus-4-6
115. **[V]** Anthropic. *Extended thinking* (Claude platform docs). https://platform.claude.com/docs/en/build-with-claude/extended-thinking

## 8. Model and pricing sources (checked 2026-11-06)

116. **[V]** OpenAI. *API Pricing.* https://developers.openai.com/api/docs/pricing
117. **[V]** DeepSeek. *Models & Pricing.* https://api-docs.deepseek.com/quick_start/pricing
118. **[V]** Vantage / AWS. *p4de.24xlarge pricing and specs* (8x A100-80GB; $27.447/hr on-demand us-east-1; spot from ~$13.98/hr). https://instances.vantage.sh/aws/ec2/p4de.24xlarge
119. **[V]** AWS. *EC2 On-Demand Instance Pricing.* https://aws.amazon.com/ec2/pricing/on-demand
120. **[V]** AWS. *AWS Cloud Credit for Research* (student awards up to $5,000). https://aws.amazon.com/government-education/research-and-technical-computing/cloud-credit-for-research
121. **[V]** AWS. *AWS Activate* (startup credits; Founders tier $1,000, portfolio tiers higher). https://aws.amazon.com/activate

---

### Notes on use

- The confirmatory program cites sections 1–5; section 6 grounds the serving
  engineering standard; sections 7–8 pin the model matrix and cost planning.
- Pricing entries are point-in-time: re-verify at probe time and update the
  `PriceTable` with a fresh `verified_at` date rather than editing history.
- Papers marked [R] that we did not independently re-open during the v0.7
  upgrade keep their repository-vetted status; a reviewer who finds any entry
  wrong should file a correction — the list is append-only by design.
