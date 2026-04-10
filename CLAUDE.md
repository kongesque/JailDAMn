# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JailDAM (Jailbreak Detection with Adaptive Memory) is a research framework for detecting jailbreak attacks against Vision-Language Models (VLMs). It uses policy-driven unsafe knowledge representations in a memory network with test-time adaptive updates. Paper accepted at COLM 2025 (arXiv:2504.03770).

## Environment Setup

```bash
conda env create -f environment.yml
conda activate llava
```

Python 3.11 with PyTorch 2.2.0 (CUDA 12.1). No pip requirements.txt — all dependencies are in `environment.yml`.

### Actual Runtime Environment (verified 2026-04-09)

**No conda available** on this machine. Packages installed directly via pip into the system Python:

```bash
pip3 install transformers openai-clip scikit-learn accelerate datasets sentence-transformers
```

| Component | Version |
|-----------|---------|
| Python | 3.11.10 |
| PyTorch | 2.4.1+cu124 |
| CUDA | 12.4 |
| GPU | NVIDIA GeForce RTX 3090 (1×) |
| transformers | 5.5.1 |
| openai-clip | 1.0.1 |
| scikit-learn | 1.8.0 |
| Jupyter | 4.2.5 (JupyterLab) |

**Note:** PyTorch version differs from `environment.yml` (2.2.0 pinned) — 2.4.1 is installed and works.

## Running the Pipeline

Three entrypoints are available:

| Script | Purpose |
|--------|---------|
| `demo.ipynb` | Original interactive pipeline (Jupyter) |
| `run_full.py` | CLI version mirroring the notebook: MM-Vet (safe) + MM-SafetyBench + FigStep + JailbreakV-Nano |
| `run_paper_eval.py` | Paper-faithful eval: 30 concepts/category, 3-way safe split, subsampled unsafe (~528) and safe test (~218) |

Run CLI scripts from the repo root:

```bash
python run_full.py
python run_paper_eval.py
```

Both print per-dataset and overall AUROC / AUPR / F1 / Precision / Recall plus latency.

### Dataset Requirements (must be present before running)

All loaders expect data under `/data/`:

| Dataset | Required paths |
|---------|---------------|
| JailbreakV-28k (`datasets/jailbreakv_28k.py`) | `/data/jailbreakv_28k/` (JSON + images) |
| JailbreakV-Nano (`datasets/jailbreakv_nano.py`) | `/data/jailbreakv_nano/jailbreakv_nano/` (JSON + images) |
| MM-SafetyBench (`datasets/mmsafety.py`) | `/data/mmsafety/` with `imgs/` and `unsafe_input/` subdirs |
| FigStep (`datasets/figstep.py`) | `/data/fig_step/` (JSON + images) |
| MM-Vet (`datasets/mmvet.py`) | `/data/mm-vet/sample.json` and `/data/mm-vet/images/` |

Missing datasets produce 0 samples; `DataLoader` will raise `ValueError: num_samples should be a positive integer value`.

`run_full.py` and `run_paper_eval.py` use MM-Vet + MM-SafetyBench + FigStep + JailbreakV-**Nano** (not 28k). JailbreakV-28k is only used via `demo.ipynb` / the old standalone loader.

## Architecture

**Core data flow:**

```
Image + Text
  → CLIP embeddings (768-dim each, concatenated to 1536-dim)
  → MemoryNetwork: soft attention over 1300 unsafe concept embeddings
  → Autoencoder: encodes attention features, computes reconstruction error
  → Decision: high error → Unsafe, low error → Safe
  → Test-time adaptation: updates least-used concept embeddings with residuals
```

**Key files:**

- `memory_network.py` — `MemoryNetwork` class: CLIP integration, soft attention over concept embeddings, 2-layer MLP classifier. This is the core module.
- `concept.json` — 1300 unsafe concept embeddings across 14 harm categories (illegal activity, hate speech, fraud, physical harm, etc.). These seed the memory network.
- `demo.ipynb` — Original interactive pipeline: dataset loading → autoencoder training on safe data → inference with threshold-based detection.
- `run_full.py` — CLI version of the full evaluation (MM-Vet + MM-SafetyBench + FigStep + JailbreakV-Nano).
- `run_paper_eval.py` — Paper-faithful CLI eval with proper train/val/test splits and subsampling to match paper numbers.
- `utils.py` — Metric computation (AUROC, AUPR, F1, precision, recall) and shared utilities.
- `ShieldEval.py` — Evaluation wrapper for the JailDAM-D defense variant, which prepends a defense prompt to queries detected as unsafe.
- `generate_dataloader.py` — Dataset preprocessing and CLIP feature extraction shared across dataset loaders.
- `SafeVLMDataset_MMsafety.py` — Standalone safe-split loader for MM-SafetyBench.

**Dataset loaders** (under `datasets/` package, each has a `main()` function):
- `datasets/jailbreakv_28k.py` — JailbreakV-28k (`UnsafeVLMDataset_28k`)
- `datasets/jailbreakv_nano.py` — JailbreakV-Nano (`UnsafeVLMDataset_jailbreakv_nano`)
- `datasets/mmsafety.py` — MM-SafetyBench (`UnsafeVLMDataset_MMsafety`)
- `datasets/figstep.py` — FigStep (`UnsafeVLMDataset_fig_step`)
- `datasets/mmvet.py` — MM-Vet safe/benign data (`VLMDataset_mmvet`)

**Detection mechanism:** The autoencoder is trained exclusively on safe data. At inference, unsafe inputs produce high reconstruction error because they contain patterns not seen during training. The adaptive memory update replaces the least-frequently-used concept embedding with the residual from the current unsafe input, allowing the system to generalize to novel jailbreak strategies without retraining.

**JailDAM-D (defense variant):** If detection fires, a defense prompt is prepended to the original VLM query before passing it to the downstream model. This is handled in `ShieldEval.py`.
