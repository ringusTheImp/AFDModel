# WX-AFD: Louisville Area Forecast Discussion Generator
## Product Requirements Document — Fine-Tuning & Evaluation

**Version:** 1.1
**Date:** February 15, 2026
**Status:** Ready for Implementation
**Authors:** Internal Working Document

---

## 1. Problem Statement

NWS forecasters at WFO Louisville (LMK) write Area Forecast Discussions (AFDs) multiple times daily — translating raw numerical weather prediction data into structured English prose. This is skilled, time-consuming labor. No open-source model exists that generates AFD-style meteorological text from structured weather input data. We're building the first one.

**Goal:** Fine-tune a language model that accepts structured weather observations/forecast data as input and produces AFD-quality meteorological prose as output, trained exclusively on Louisville WFO products.

---

## 2. What We Already Have

| Asset | Details |
|-------|---------|
| **Training data** | 2,408 prompt-completion pairs (2024-01-01 → 2025-01-01) |
| **Split** | 2,287 train / 121 validation |
| **Input format** | Structured weather data: 3-day, 12-hourly trajectories at 38.25°N, 85.76°W (~424 tokens avg) |
| **Output format** | AFD text from WFO Louisville/LMK (~1,555 tokens avg) |
| **Total tokens** | ~4.8M training tokens |
| **Location** | `/mnt/user-data/outputs/wx-dataset-derecho/` |
| **Files** | `data/train.jsonl`, `data/val.jsonl` |
| **Compute** | NCAR Derecho — A100 40GB GPU nodes |

---

## 3. The Stack (Final — No Alternatives)

We're committing to this. No hedging.

### 3.1 Base Model: Qwen3-4B-Instruct-2507

[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) — Alibaba, July 2025, Apache 2.0.

**Why this model, specifically:**

- **Non-thinking-only variant.** No `<think>` blocks to manage during training — clean ChatML input/output. The original Qwen3-4B dual-mode model requires empty `<think>\n\n</think>\n\n` prefixes in every assistant response; the `-2507` variant eliminates this entirely.
- **#1 on distillabs fine-tuning benchmarks** (Feb 2026). Not #1 at raw inference — #1 at *learning from fine-tuning data*. That's what matters for us.
- **Architecture:** 36-layer dense transformer, 2,560 hidden dim, GQA (32 Q / 8 KV heads), SwiGLU, RMSNorm, RoPE. 262K native context window (we'll use ~2K). 151,936 vocab size, bfloat16.
- **36 trillion training tokens, 119 languages.** The base knowledge is massive — we're teaching it *style and format*, not meteorology from scratch.
- **Requires:** `transformers >= 5.0.0`

### 3.2 Fine-Tuning Framework: Axolotl v0.14.0

[Axolotl](https://github.com/axolotl-ai-cloud/axolotl) (11k+ stars, Apache 2.0, released January 30, 2026).

**Why Axolotl over TRL/Unsloth/torchtune:**

- **First-class Qwen3 support** — `chat_template: qwen3` works out of the box. [Qwen's own docs link to Axolotl](https://qwen.readthedocs.io/en/latest/training/axolotl.html) as an official fine-tuning path.
- **Sample packing (multipacking)** — critical for our dataset. Our inputs average ~424 tokens and outputs ~1,555 tokens. Without packing, most of every 2K-token batch is padding. Axolotl's multipacking bins multiple examples into single sequences, dramatically improving GPU utilization. With 2,408 short-to-medium examples, this alone can cut training time 40-60%.
- **Loss masking built-in** — `roles_to_train: [assistant]` in YAML. No DataCollator gymnastics, no token ID debugging, no [ZeroDivisionError from template matching failures](https://github.com/unslothai/unsloth/issues/2771).
- **YAML-driven configs** — fully reproducible. The entire training run is defined in one file. No Python scripts to maintain.
- **Transformers v5 compatible** — v0.14.0 shipped with [transformers v5 support](https://github.com/axolotl-ai-cloud/axolotl/releases) and CUDA 13.0 compatibility.
- **Active maintenance** — multiple releases in Jan/Feb 2026. Compare to [torchtune, which stopped active development July 2025](https://github.com/meta-pytorch/torchtune/issues/2883).

**What we're NOT using and why:**

- **Unsloth:** Kernel-level speed optimizer. Doesn't change training methodology — just makes HF Trainer faster. Disables speed optimizations when `lora_dropout > 0`, which we need for regularization on a small dataset. The speed gain isn't worth the constraints.
- **TRL SFTTrainer:** Solid, but Axolotl wraps it with better data handling, packing, and config management. We'd end up reimplementing half of Axolotl's features manually.
- **torchtune:** [Dead as of July 2025](https://github.com/meta-pytorch/torchtune/issues/2883). Successor repo doesn't exist yet. The code works but no bug fixes are coming.
- **Bare PyTorch:** Romantic but impractical. Manually handling ChatML tokenization, loss masking, gradient accumulation, and mixed precision for a production fine-tune is weeks of work Axolotl already solved.

### 3.3 Adaptation Method: DoRA + rsLoRA (Rank 16)

**DoRA (Weight-Decomposed Low-Rank Adaptation)** decomposes weight updates into magnitude and direction components. The [original DoRA paper](https://arxiv.org/abs/2402.09353) (ICML 2024 Oral) showed DoRA outperforms LoRA even at half the rank, and [DoRAN](https://arxiv.org/abs/2510.04331) (2025) further stabilizes the approach via noise injection.

**rsLoRA (Rank-Stabilized LoRA)** scales adapter output by `lora_alpha / sqrt(r)` instead of `lora_alpha / r`. Zero overhead, stabilizes training at any rank. No reason not to use it — it's [strictly better than standard LoRA scaling](https://arxiv.org/abs/2312.03732).

**Why rank 16, not 8 or 32:**

- Rank 8 with DoRA roughly matches rank 32 vanilla LoRA. We want headroom above that baseline.
- Rank 32 on 2,408 examples risks overfitting — more adapter capacity than our data can fill.
- Rank 16 with DoRA is the sweet spot: enough capacity to learn Louisville AFD style without memorizing training examples.

**Why NOT full fine-tuning:** [Schulman (Sept 2025)](https://thinkingmachines.ai/blog/lora/) — "LoRA Without Regret" (technical report) — showed LoRA performs identically to full fine-tuning on small-to-medium instruction datasets. Full FT on 2,408 examples risks severe overfitting with 3.7B trainable parameters. LoRA constrains us to ~27M trainable params (0.7% of model). That's a feature, not a limitation.

### 3.4 Compute: NCAR Derecho

| Resource | Spec |
|----------|------|
| **GPU** | NVIDIA A100 40GB (4 per node, NVLink 600 GB/s) |
| **CPU** | 64× AMD Milan cores per node |
| **RAM** | 512 GB system memory per node |
| **Access** | Direct login to compute nodes |
| **CUDA** | 12.2.1 (via `module load cuda/12.2.1`) |

Single-GPU training. Qwen3-4B in 4-bit QLoRA uses ~4.2 GB VRAM — leaves ~36 GB headroom on A100 40GB. We could run 16-bit LoRA (~10 GB) with room to spare. We'll use **bf16 LoRA (not quantized)** since we have the VRAM budget.

---

## 4. Data Format

### 4.1 Prompt-Completion Structure

Axolotl's `chat_template` type with ChatML. Each example is a single-turn conversation:

```json
{
  "messages": [
    {"role": "system", "content": "<see wx_afd.SYSTEM_PROMPT>"},
    {"role": "user", "content": "WEATHER DATA FOR LOUISVILLE, KY (38.25°N, 85.76°W)\nValid: 2024-03-15T12:00:00Z\n\n[structured weather trajectory data...]"},
    {"role": "assistant", "content": "AREA FORECAST DISCUSSION\nNATIONAL WEATHER SERVICE LOUISVILLE KY\n[AFD text...]"}
  ]
}
```

The canonical system prompt lives in `wx_afd.SYSTEM_PROMPT` and is used by `03_build_dataset.py`, the training notebook, and all inference/evaluation scripts.

### 4.2 Dataset Format

`03_build_dataset.py` outputs `data/train.jsonl` and `data/val.jsonl` directly in Axolotl's `messages` format. No separate conversion step is needed.

### 4.3 Chat Template

Qwen3 uses ChatML. Axolotl handles this via `chat_template: qwen3`. The rendered format:

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{weather_data}<|im_end|>
<|im_start|>assistant
{afd_text}<|im_end|>
```

### 4.4 Critical Token Configuration

**This is the single most dangerous pitfall in the entire pipeline.**

| Token | ID | Purpose | Gotcha |
|-------|----|---------|--------|
| `<\|im_end\|>` | 151645 | **EOS token** — signals end of generation | A tokenizer update swapped the default. If not set explicitly, model generates endlessly. |
| `<\|endoftext\|>` | 151643 | **PAD token** | If this collides with EOS, model never learns to stop. Must be distinct from EOS. |
| `<\|im_start\|>` | 151644 | Turn delimiter | Axolotl handles this via chat template. |

**In the Axolotl config, we explicitly set:**
```yaml
special_tokens:
  eos_token: "<|im_end|>"
  pad_token: "<|endoftext|>"
```

This is non-negotiable. Skip this and the model will generate infinite text at inference. There is no warning — it just silently breaks.

---

## 5. Training Configuration

### 5.1 Axolotl YAML Config

This is the complete, production-ready configuration file:

```yaml
# ============================================================
# WX-AFD: Louisville AFD Generator — Axolotl Training Config
# Model: Qwen3-4B-Instruct-2507
# Method: DoRA + rsLoRA (rank 16), bf16
# Data: 2,287 train / 121 val examples
# Compute: NCAR Derecho A100 40GB (single GPU)
# ============================================================

base_model: Qwen/Qwen3-4B-Instruct-2507
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer
trust_remote_code: true

# Chat template
chat_template: qwen3

# Token overrides (CRITICAL — see Section 4.4)
special_tokens:
  eos_token: "<|im_end|>"
  pad_token: "<|endoftext|>"

# ---- Adapter Configuration ----
adapter: lora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_linear: true          # All linear layers (q,k,v,o,gate,up,down)
peft_use_rslora: true             # Rank-stabilized scaling
peft_use_dora: true               # Weight-decomposed LoRA

# ---- Dataset ----
datasets:
  - path: /glade/derecho/scratch/$USER/wx-afd/data/train.jsonl
    type: chat_template
    roles_to_train:
      - assistant                 # Loss only on AFD output, not weather input
    train_on_eos: true            # Learn to stop generating

val_set_size: 0                   # We use a pre-split val set
test_datasets:
  - path: /glade/derecho/scratch/$USER/wx-afd/data/val.jsonl
    type: chat_template
    roles_to_train:
      - assistant

# End-of-turn tokens (global, not per-dataset — per Axolotl docs)
eot_tokens:
  - "<|im_end|>"

# ---- Training Hyperparameters ----
num_epochs: 3
micro_batch_size: 2               # Per-GPU batch size
gradient_accumulation_steps: 8    # Effective batch size: 2 × 8 = 16
learning_rate: 1.5e-4             # 10× higher than full FT (per Schulman 2025)
lr_scheduler: cosine
warmup_ratio: 0.05                # ~5% warmup
weight_decay: 0.01
optimizer: adamw_torch
max_grad_norm: 1.0

# ---- Precision & Performance ----
bf16: true
tf32: true
gradient_checkpointing: true
sample_packing: true              # Bin multiple examples per sequence
pad_to_sequence_len: true
sequence_len: 2048                # Covers 99%+ of our examples (~1,979 avg total)

# ---- Saving & Evaluation ----
output_dir: /glade/derecho/scratch/$USER/wx-afd/output
save_strategy: steps
save_steps: 100
save_total_limit: 5
eval_strategy: steps
eval_steps: 100
logging_steps: 10

# ---- Early Stopping ----
early_stopping_patience: 5        # Stop if val loss doesn't improve for 5 evals
load_best_model_at_end: true
metric_for_best_model: eval_loss

# ---- Reproducibility ----
seed: 42

# ---- Derecho-specific ----
flash_attention: true             # A100 supports FlashAttention-2
```

### 5.2 Hyperparameter Rationale

| Parameter | Value | Why |
|-----------|-------|-----|
| **Learning rate: 1.5e-4** | 10× higher than full FT | [Schulman 2025](https://thinkingmachines.ai/blog/lora/) found LoRA tolerates and benefits from higher LR than full fine-tuning on small datasets |
| **Effective batch size: 16** | 2 × 8 accumulation | LoRA is less tolerant of large batches than full FT. 16 balances gradient noise with stable convergence. |
| **Epochs: 3 w/ early stopping** | Multi-epoch can degrade | More than 2-3 epochs on small instruction datasets degrades instruction-following ability. Early stopping catches this. |
| **Dropout: 0.05** | Regularization for small data | 2,408 examples is small. Even mild dropout helps prevent memorization. |
| **Sequence length: 2048** | Covers our distribution | Mean total tokens: ~1,979. 2048 covers 99%+ without wasting memory on empty space. |
| **Sample packing: true** | GPU efficiency | Short examples (~424 token inputs) waste massive padding without packing. Axolotl's multipacking fills sequences efficiently. |
| **Warmup: 5%** | Standard cosine warmup | ~34 warmup steps on our dataset. Prevents early instability. |

### 5.3 What NOT to Do

These are real failure modes we've identified during research. Each one will silently break training:

1. **Don't use greedy decoding at inference.** Causes performance degradation and repetition loops. Always use sampling (temperature 0.7, top_p 0.9) or beam search.
2. **Don't forget `eos_token: "<|im_end|>"`** — model generates forever without it.
3. **Don't use `packing: true` with `DataCollatorForCompletionOnlyLM`** — they're incompatible. Axolotl's native `roles_to_train` handles this correctly.
4. **Don't pass response template as string** — BPE context sensitivity means tokenizing `"<|im_start|>assistant\n"` as a string produces different token IDs than the same bytes in context. Axolotl's chat template approach avoids this entirely.
5. **Don't include `<think>` blocks** — we're using the non-thinking `-2507` variant. Training data should contain plain assistant text only.

---

## 6. Implementation Plan

### Phase 1: Environment Setup (Day 1)

**On Derecho login node:**

```bash
# Load base modules
module purge
module load ncarenv/23.09 cuda/12.2.1
module load conda

# Create environment
CONDA_OVERRIDE_CUDA=12.1 conda create -n wx-afd python=3.11 -y
conda activate wx-afd

# Install stack
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install axolotl==0.14.0
pip install flash-attn --no-build-isolation

# Verify
python -c "import torch; print(torch.cuda.is_available())"
python -c "import axolotl; print(axolotl.__version__)"
```

**Storage layout:**

```
/glade/derecho/scratch/$USER/wx-afd/
├── wx_afd.py                    # Shared constants & utilities
├── data/
│   ├── train.jsonl              # Training data (messages format)
│   └── val.jsonl                # Validation data (messages format)
├── configs/
│   └── wx-afd-dora.yml          # Axolotl config (Section 5.1)
├── output/                      # Checkpoints, logs
│   ├── checkpoint-100/
│   ├── checkpoint-200/
│   └── ...
├── eval/                        # Evaluation outputs
│   ├── generated/               # Model-generated AFDs
│   └── scores/                  # Metric results
├── scripts/
│   ├── evaluate.py              # Evaluation pipeline
│   └── generate.py              # Inference script
└── tests/
    └── test_dataset.py          # Dataset validation checks
```

### Phase 2: Data Preparation (Day 1)

`03_build_dataset.py` already outputs `data/train.jsonl` and `data/val.jsonl` in Axolotl's messages format (system + user + assistant). No separate conversion step is needed.

### Phase 3: Sanity Check (Day 1-2)

Before running a real training job, validate everything works:

```bash
# Validate config and data loading (no training)
accelerate launch -m axolotl.cli.train configs/wx-afd-dora.yml \
    --debug \
    --num_epochs 0
```

**Checklist before proceeding:**
- [ ] Data loads without errors
- [ ] Tokenizer correctly identifies `<|im_end|>` as EOS (token 151645)
- [ ] PAD token is `<|endoftext|>` (token 151643), distinct from EOS
- [ ] Sample packing reports expected number of packed sequences
- [ ] Loss masking: verify labels are -100 for system + user tokens
- [ ] GPU memory usage is within expected range (~10 GB for bf16 LoRA)
- [ ] FlashAttention-2 is detected and active

### Phase 4: Training (Days 2-3)

```bash
# Launch training
accelerate launch -m axolotl.cli.train configs/wx-afd-dora.yml
```

**Expected timeline:**
- ~2,287 examples ÷ effective batch 16 = ~143 steps/epoch
- 3 epochs = ~429 total steps
- With packing and bf16 on A100: ~15-25 minutes per epoch
- **Total: ~45-75 minutes of GPU time**
- Early stopping may terminate after epoch 2

The checkpoint saves every 100 steps protect against interruption. If training is interrupted, resume from latest checkpoint:

```bash
accelerate launch -m axolotl.cli.train configs/wx-afd-dora.yml \
    --resume_from_checkpoint /glade/derecho/scratch/$USER/wx-afd/output/checkpoint-200
```

### Phase 5: Merge & Export (Day 3)

After training, merge the LoRA adapter back into the base model for clean inference:

```bash
# Merge adapter into base model
accelerate launch -m axolotl.cli.merge_lora configs/wx-afd-dora.yml \
    --lora_model_dir /glade/derecho/scratch/$USER/wx-afd/output

# Copy to persistent storage
cp -r /glade/derecho/scratch/$USER/wx-afd/output/merged \
      /glade/work/$USER/wx-afd-model-v1/
```

The merged model is a standard HuggingFace model directory — loadable with `AutoModelForCausalLM.from_pretrained()` anywhere. No adapter loading required at inference.

---

## 7. Evaluation Framework

We evaluate on two axes: **text quality** (does it read like a real AFD?) and **factual consistency** (does it say things consistent with the input weather data?).

### 7.1 Evaluation Data

The 121 held-out validation examples. For each:
1. Feed the weather input to the fine-tuned model
2. Generate an AFD with `temperature=0.7, top_p=0.9, max_new_tokens=2048`
3. Compare generated AFD against the real AFD from the same timestamp

### 7.2 Baseline Comparisons

| Baseline | Description |
|----------|-------------|
| **Zero-shot Qwen3-4B-Instruct-2507** | Same model, no fine-tuning, same system prompt. Measures what fine-tuning actually bought us. |
| **Zero-shot Qwen3-4B (original dual-mode)** | Original thinking-mode model with `enable_thinking=False`. Secondary comparison. |
| **Copy-nearest-neighbor** | Find the training example with the most similar weather input, return its AFD verbatim. Sanity check — if our model can't beat this, something is broken. |

### 7.3 Automated Metrics

#### 7.3.1 Standard NLG Metrics

| Metric | What It Measures | Library |
|--------|-----------------|---------|
| **ROUGE-1/2/L** | N-gram overlap with reference AFD | [rouge-score](https://pypi.org/project/rouge-score/) |
| **BERTScore** | Semantic similarity via contextual embeddings | [bert-score](https://github.com/Tiiiger/bert_score) |

These are necessary but not sufficient. An AFD that mentions "temperatures in the mid-40s" when the data shows 72°F will score well on ROUGE if the rest matches. We need factual consistency metrics.

#### 7.3.2 Factual Consistency: AlignScore

[AlignScore](https://github.com/yuh-zha/AlignScore) (ACL 2023, 300+ stars) — a unified factual consistency metric trained on 4.7M examples. State-of-the-art on SummaC and TRUE benchmarks.

**How we use it:** Score the generated AFD against the structured weather input data. AlignScore answers: "Is the text factually consistent with the source?" This directly measures whether the model is hallucinating weather conditions.

```python
from alignscore import AlignScorer

scorer = AlignScorer(model="roberta-large", batch_size=16, device="cuda")

# For each validation example:
score = scorer.score(
    contexts=[weather_input_text],      # Source weather data
    claims=[generated_afd_text]          # Model's generated AFD
)
# Returns 0-1 score. Higher = more factually consistent.
```

### 7.4 Evaluation Pipeline

The evaluation pipeline lives in `scripts/evaluate.py` and imports shared code from `wx_afd.py` (`load_model`, `generate_afd`, `compute_rouge`, `compute_bertscore`, `format_compliance`, `SYSTEM_PROMPT`).

```bash
# See scripts/evaluate.py (imports shared code from wx_afd.py)
# Usage:
python scripts/evaluate.py \
    --model /path/to/merged/model \
    --val_data data/val.jsonl \
    --tag finetuned
```

The script generates AFDs for all validation examples, computes ROUGE-1/2/L, BERTScore, and format compliance, then saves results to `eval/<tag>/scores/metrics.json`.

### 7.5 Evaluation Commands

```bash
# Evaluate fine-tuned model
python scripts/evaluate.py \
    --model output/merged \
    --val_data data/val.jsonl \
    --tag finetuned \
    --output_dir eval/finetuned

# Evaluate zero-shot baseline (same model, no fine-tuning)
python scripts/evaluate.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --val_data data/val.jsonl \
    --tag zero-shot \
    --output_dir eval/zero-shot
```

### 7.6 Success Criteria

| Metric | Minimum Acceptable | Target | Notes |
|--------|-------------------|--------|-------|
| **ROUGE-L vs zero-shot** | +0.05 | +0.10 | Fine-tuning must meaningfully improve over prompting |
| **BERTScore F1** | > 0.70 | > 0.80 | Semantic similarity to real AFDs |
| **AlignScore** | > 0.60 | > 0.75 | Factual consistency with input data |
| **Temperature MAE** | < 5°F | < 3°F | Extracted temps vs source data |
| **EOS behavior** | 100% | 100% | Model must stop generating (no infinite loops) |
| **Format compliance** | > 80% | > 95% | Correct AFD section headers, structure |

Literature on LoRA fine-tuning of 4B-class models with ~2,400 domain-specific examples suggests ROUGE-L improvements of +0.07–0.12 are achievable, placing our +0.05 minimum / +0.10 target within realistic bounds.

If ROUGE-L doesn't improve over zero-shot, the fine-tuning failed and we investigate data quality before retraining.

---

## 8. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **EOS token misconfiguration** | High (default is wrong) | Critical — infinite generation | Explicit `eos_token` in config + sanity check in Phase 3 |
| **Overfitting (memorization)** | Medium | Model regurgitates training examples | DoRA rank 16 + dropout 0.05 + early stopping + eval on held-out set |
| **Catastrophic forgetting** | Low | Model loses general language ability | LoRA constrains updates to 0.7% of params; base weights frozen |
| **Training interrupted** | Low | Lost training progress | Checkpoint every 100 steps + resume support |
| **Hallucinated weather data** | Medium | Dangerous in operational context | AlignScore eval + entity extraction + human review |
| **Sample packing label corruption** | Low | Wrong tokens trained on | Axolotl's native packing handles label alignment; verify in sanity check |
| **Qwen3-4B-Instruct-2507 model update** | Low | Breaking changes | Pin to specific HF revision hash in config |

---

## 9. Future Work (Out of Scope for v1)

These are noted for future iterations. None of these are in the current scope:

- **Weather Entity Extraction & Comparison** *(highest-priority eval improvement for v2)* — Extract specific meteorological claims (temperatures, PoP, wind, precipitation type, sky condition, timing references) from both generated and reference AFDs, then compare against source data. Score via temperature MAE (°F), PoP MAE (%), categorical exact match, and temporal consistency. This is the metric that matters most for operational use — a model that writes beautiful prose but gets temperatures wrong by 15°F is useless.
- **Multi-WFO training** — Expand beyond Louisville to 10+ WFOs. Requires dataset pipeline changes.
- **Attribute-specific adapters** — [Met Office VLM study (Dec 2025)](https://arxiv.org/abs/2512.03623) found separate models for wind/precip/visibility outperform a single model. Consider separate LoRA adapters for different AFD sections (synopsis, short-term, aviation).
- **GRPO/DPO alignment** — Use [AlignScore](https://github.com/yuh-zha/AlignScore) as a reward signal to construct preference pairs. Fine-tune with DPO to penalize hallucination. Axolotl supports this natively.
- **Cycle consistency training** — [Amazon Science (ACL 2023)](https://www.amazon.science/code-and-datasets/faithful-low-resource-data-to-text-generation-through-cycle-training) showed data→text→data cycle training works with ~100 examples. Train an inverse model (AFD→structured data) and use cycle loss for faithfulness.
- **Curriculum learning** — Order training examples by difficulty (shorter/simpler AFDs first, complex multi-hazard events last). [Difficulty-aware bucketed fine-tuning (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0950705125016946) showed zigzag progression outperforms strict ordering.
- **Custom tokenizer extension** — Add weather-specific tokens (METAR codes, station IDs) via `tokenizer.add_tokens()` + [smart initialization from sub-word constituents](https://medium.com/everyday-ai/add-extra-new-tokens-to-pre-trained-llm-tokenizer-05b0a89f058f).
- **Inference serving** — vLLM or TGI deployment for real-time generation.
- **Weather text evaluation framework** — No dedicated tool exists. Build one using keyword extraction + entity comparison + [AlignScore](https://github.com/yuh-zha/AlignScore). The [Hierarchical AI-Meteorologist](https://arxiv.org/abs/2511.23387) keyword validation approach is a starting point.

---

## 10. Timeline

| Day | Phase | Deliverable |
|-----|-------|-------------|
| **1** | Environment setup + dataset validation | Working conda env on Derecho, validated dataset |
| **1-2** | Sanity check | All pre-flight checks passing |
| **2-3** | Training | Trained model with early stopping |
| **3** | Merge & export | Merged model in persistent storage |
| **3-4** | Evaluation | Metrics for fine-tuned + zero-shot baseline |
| **4** | Analysis | Results comparison, failure case review |

**Total: ~4 working days from start to evaluated model.**

---

## 11. Key References

| Resource | URL |
|----------|-----|
| Qwen3-4B-Instruct-2507 | https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 |
| Axolotl (GitHub) | https://github.com/axolotl-ai-cloud/axolotl |
| Axolotl Docs | https://docs.axolotl.ai/ |
| Qwen3 × Axolotl Guide | https://qwen.readthedocs.io/en/latest/training/axolotl.html |
| DoRA Paper | https://arxiv.org/abs/2402.09353 |
| DoRAN Paper | https://arxiv.org/abs/2510.04331 |
| rsLoRA Paper | https://arxiv.org/abs/2312.03732 |
| Schulman — LoRA Without Regret | https://thinkingmachines.ai/blog/lora/ |
| AlignScore (eval) | https://github.com/yuh-zha/AlignScore |
| Met Office VLM Study | https://arxiv.org/abs/2512.03623 |
| Hierarchical AI-Meteorologist | https://arxiv.org/abs/2511.23387 |
| NCAR Derecho Docs | https://arc.ucar.edu/knowledge_base/74317833 |
| IEM NWS Text Archives | https://mesonet.agron.iastate.edu/nws/text.php |
| Loss Truncation | https://github.com/ddkang/loss_dropper |
| Cycle Training (Amazon) | https://www.amazon.science/code-and-datasets/faithful-low-resource-data-to-text-generation-through-cycle-training |
| torchtune Sunset Notice | https://github.com/meta-pytorch/torchtune/issues/2883 |

---

## Changelog

### v1.1 (2026-02-15)

- **Consolidated shared code** into `wx_afd.py`: `SYSTEM_PROMPT`, `REQUIRED_SECTIONS`, `load_jsonl`, `load_model`, `generate_afd`, `compute_rouge`, `compute_bertscore`, `format_compliance`
- **Fixed hardcoded token IDs** (151645/151643) — now resolved dynamically via `tokenizer.convert_tokens_to_ids()`
- **Updated PRD** to match actual implementation: corrected YAML config (`test_datasets`, `eot_tokens`, filenames), removed obsolete `convert_data.py` references, updated storage layout
- **Added project hygiene files**: `.gitignore`, `requirements.txt`
- **Moved `test_dataset.py`** to `tests/` directory
- **Updated `setup_derecho.sh`** to copy `wx_afd.py` to Derecho
