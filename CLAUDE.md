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

The complete training and evaluation pipeline lives in `demo.ipynb`. Run it with Jupyter after activating the conda environment. There is no CLI entrypoint or test runner — validation is done through metrics (AUROC, AUPR, F1) computed within the notebook.

### Dataset Requirements (must be present before running)

The notebook fails at cell 1 if datasets are missing. All four loaders expect data under `/data/`:

| Loader | Required path |
|--------|--------------|
| `UnsafeVLMDataset_28k.py` | `/data/jailbreakv_28k/` (JSON + images) |
| `UnsafeVLMDataset_MMsafety.py` | `/data/mmsafety/imgs/` and `/data/mmsafety/unsafe_input/` |
| `UnsafeVLMDataset_fig_step.py` | `/data/fig_step/` (check loader for exact path) |
| `VLMDataset_mmvet.py` | `/data/mm-vet/sample.json` and corresponding images |

Without these, all datasets load 0 samples and `DataLoader` raises `ValueError: num_samples should be a positive integer value`.

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
- `demo.ipynb` — Full pipeline: dataset loading → autoencoder training on safe data → inference with threshold-based detection.
- `utils.py` — Metric computation (AUROC, AUPR, F1, precision, recall) and shared utilities.
- `ShieldEval.py` — Evaluation wrapper for the JailDAM-D defense variant, which prepends a defense prompt to queries detected as unsafe.
- `generate_dataloader.py` — Dataset preprocessing and CLIP feature extraction shared across dataset loaders.

**Dataset loaders** (each has a `main()` function called from the notebook):
- `UnsafeVLMDataset_28k.py` — JailbreakV-28k
- `UnsafeVLMDataset_MMsafety.py` — MM-SafetyBench
- `UnsafeVLMDataset_fig_step.py` — FigStep
- `VLMDataset_mmvet.py` — MM-Vet (safe/benign data)

**Detection mechanism:** The autoencoder is trained exclusively on safe data. At inference, unsafe inputs produce high reconstruction error because they contain patterns not seen during training. The adaptive memory update replaces the least-frequently-used concept embedding with the residual from the current unsafe input, allowing the system to generalize to novel jailbreak strategies without retraining.

**JailDAM-D (defense variant):** If detection fires, a defense prompt is prepended to the original VLM query before passing it to the downstream model. This is handled in `ShieldEval.py`.
