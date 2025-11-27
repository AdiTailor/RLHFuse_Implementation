# RLHFuse on IMDb with DistilGPT2

This project is a small‑scale implementation of the ideas from:

> Zhong et al., “Optimizing RLHF Training for Large Language Models with Stage Fusion”, NSDI 2025.

The goal is to demonstrate how **inter‑stage** and **intra‑stage fusion** can improve the efficiency of RLHF training on modest hardware (2 GPUs), using a sentiment‑generation task on the IMDb Reviews dataset and a lightweight DistilGPT2 model.

---

## 1. Overview

- **Base model:** `distilgpt2` (~124M parameters)
- **Task:** Sentiment‑aware text generation on IMDb movie reviews
- **Dataset:** IMDb Reviews (50k reviews, balanced positive/negative)
- **Hardware (used in experiments):**
  - 2 × NVIDIA GeForce RTX 4060 Ti
  - Mixed precision + multi‑GPU via Hugging Face Accelerate
- **Training stages:**
  1. Supervised Fine‑Tuning (SFT) with LoRA adapters  
  2. Reward model training on preference pairs  
  3. PPO‑based RLHF with RLHFuse‑style stage fusion
- **Key results (example run):**
  - Average reward after RLHF: **≈ 3.36**
  - Average throughput: **≈ 5.1–5.3 samples/sec**
  - End‑to‑end speedup vs. synthetic DeepSpeed‑Chat baseline: **≈ 2.8×**
  - Peak GPU memory allocated: **≈ 2.5 GB** (reserved ~3.7 GB, stable)

All logic is contained in:

- `run_rlhf_baseline.py`

The script also generates figures such as:

- `training_metrics.jpg`
- `summary_dashboard.jpg`
- `gpu_memory_usage.jpg`
- `throughput_over_time.jpg`
- `speedup_comparison.jpg`
- `stage_breakdown.jpg`

---

## 2. RLHF Pipeline

The project implements a full RLHF pipeline inspired by the original RLHFuse system:

### 2.1 Stage 1 – Supervised Fine‑Tuning (SFT)

- **Model:** DistilGPT2 with LoRA adapters (`LoraConfig`, `get_peft_model`)
- **Data:** ~4,000 IMDb reviews
- **Training setup:**
  - 3 epochs
  - LoRA rank `r = 8`
- **Goal:** Teach the model to generate coherent review‑style text conditioned on an input review prompt.

### 2.2 Stage 2 – Reward Model

- **Architecture:** DistilGPT2 backbone with a scalar reward head
- **Data:**
  - ~8,000 IMDb samples
  - ~2,000 preference pairs (better/worse continuation)
- **Objective:** Learn a scalar reward `R_ϕ(x, y)` that scores how good a generated continuation `y` is for review `x`.  
  Outputs are passed through `tanh` to keep rewards bounded.

### 2.3 Stage 3 – PPO RLHF

- **Policy (actor):** SFT model
- **Value function (critic):** Value head on top of the policy
- **Training setup:**
  - ~40 PPO iterations
  - 3 PPO epochs per iteration
  - Generalized Advantage Estimation (GAE)
  - Adaptive KL coefficient
  - Temperature annealing
  - Simple curriculum over prompts

The PPO loop is instrumented with detailed metrics and visualizations.

---

## 3. RLHFuse‑Style Optimizations

The project brings RLHFuse’s system ideas to a 2‑GPU environment.

### 3.1 Inter‑Stage Fusion

**Goal:** Overlap generation, inference, and reward computation to reduce idle time.

Implemented mechanisms:

- **Sample‑level subtask interleaving:**  
  Generation, reward scoring, and training are not treated as monolithic blocks; instead, mini‑batches are pipelined so later stages start as soon as enough samples are ready.
- **Hidden state caching (`HiddenStateCache`):**  
  Stores prefix hidden states (up to 4096 entries) to avoid recomputing them when prompts share common prefixes, reducing generation cost.

### 3.2 Intra‑Stage Fusion

**Goal:** Reduce pipeline “bubbles” within PPO training.

Implemented mechanisms:

- **Microbatch fused schedule:**  
  Actor and critic updates are coordinated so that forward and backward passes overlap where possible, mimicking RLHFuse intra‑stage fusion at small scale.
- **Stabilization tricks:**
  - Adaptive KL penalty to keep the policy close to the reference model
  - Temperature annealing
  - GAE for low‑variance advantages

Together, these yield an observed **~2.8× throughput improvement** over a synthetic DeepSpeed‑Chat baseline (computed from our throughput).

---

## 4. Installation

### 4.1 Requirements

- Python ≥ 3.9
- CUDA‑enabled PyTorch
- Hugging Face stack and plotting libraries

Install dependencies (example):

pip install
torch
transformers
datasets
accelerate
peft
numpy
tqdm
matplotlib
seaborn

Make sure your environment sees your GPUs, e.g.:
export CUDA_VISIBLE_DEVICES=0,1

--

## 5. Usage

Run the full pipeline (SFT → reward model → PPO with fusion):
python run_rlhf_baseline.py


What this does:

1. Loads the IMDb dataset (50k movie reviews, balanced pos/neg).
2. Performs LoRA‑based SFT on a subset of reviews.
3. Trains a reward model on preference pairs.
4. Runs PPO with RLHFuse‑style inter‑ and intra‑stage fusion.
5. Logs metrics and writes plots into a `figs/` directory (or the directory configured in the script).

You can customize hyperparameters (dataset size, number of PPO iterations, LoRA config, etc.) by editing the constants in `run_rlhf_baseline.py`.

---

## 7. Metrics & Visualizations

The script logs metrics via a `PerformanceMetricsTracker` and generates multiple figures.

### 7.1 `training_metrics.jpg`

Four panels:

- **Reward Progression:**  
  Average reward vs iteration. In our runs, the reward stabilizes around 3.3–3.6.
- **Actor Loss Over Time:**  
  PPO policy loss; generally trends downward, then oscillates at a stable level.
- **Critic Loss Over Time:**  
  Value loss; convergence indicates a good value function approximation.
- **Reward Distribution:**  
  Histogram of reward values with a vertical line at the mean (≈ 3.36).

### 7.2 `summary_dashboard.jpg` – RLHFuse Performance Dashboard

A single slide summarizing:

- **Key metrics block:**
  - Average throughput (samples/sec)
  - Total training time (minutes)
  - Total iterations
  - Average reward
  - Peak GPU memory
  - Approximate speedup vs baseline (~2.8×)
- **Throughput Trend:** Throughput vs iteration.
- **Reward Progression:** Same as in `training_metrics`.
- **GPU Memory Usage:** Allocated memory vs iteration.
- **Stage Distribution:** Pie chart of time spent in generation / inference / training.
- **Training Losses:** Actor and critic loss curves.
- **Efficiency Gains Panel:**  
  Text summary of inter‑stage fusion, intra‑stage fusion, GPU utilization, and memory efficiency (hidden‑state caching).

### 7.3 `gpu_memory_usage.jpg`

- **Left:** Allocated GPU memory vs iteration, hovering between ~2.4–2.5 GB with healthy fluctuations that track workload.
- **Right:** Reserved GPU memory vs iteration, flat at ~3.71 GB, indicating no memory leaks and stable allocation behavior.

### 7.4 `throughput_over_time.jpg`

- Throughput (samples/sec) vs iteration.
- After a warm‑up iteration, throughput stabilizes in the 5.0–5.9 samples/sec range.
- A dashed line shows the average throughput.

### 7.5 `speedup_comparison.jpg`

Two bar charts:
- **End‑to‑End Throughput Comparison:**



