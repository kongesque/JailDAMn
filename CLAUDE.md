# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

JailDAM (Jailbreak Detection with Adaptive Memory) is a research framework for detecting jailbreak attacks against Vision-Language Models (VLMs). Paper accepted at COLM 2025 (arXiv:2504.03770).

## Runtime Environment

**RunPod image:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`

PyTorch and CUDA come pre-installed in this image. Only install:

```bash
pip install transformers scikit-learn Pillow numpy
```

| Component | Version |
|-----------|---------|
| Python | 3.11 |
| PyTorch | 2.4.0+cu124 |
| CUDA | 12.4 |
| GPU | NVIDIA GeForce RTX 3090 (1x) |

No conda. No `environment.yml`. The repo is at `kongesque/JailDAMn` on GitHub (was a fork of `ShenzheZhu/JailDAM`, now standalone).

## Running

```bash
python run_paper_eval.py
```

This is the only reliable entrypoint. Prints per-dataset and overall AUROC / AUPR / F1 / Precision / Recall. `run_full.py` and `demo.ipynb` are archived — they had threshold leakage and inflated metrics.

### Dataset Setup

Datasets live under `/data/`. Download and extract before running:

| Dataset | Path |
|---------|------|
| MM-SafetyBench | `/data/mmsafety/` (needs `imgs/` and `unsafe_input/` subdirs) |
| FigStep | `/data/fig_step/` |
| JailbreakV-Nano | `/data/jailbreakv_nano/` |
| MM-Vet | `/data/mm-vet/` (needs `sample.json` and `images/`) |

Missing datasets cause `ValueError: num_samples should be a positive integer value`.

`gdown` works for downloading from Google Drive. Extraction via `python3 -c "import zipfile; zipfile.ZipFile('file.zip').extractall('/data/')"` (no `unzip` on RunPod).

## Architecture

```
Image + Text
  -> CLIP embeddings (768-dim each, concatenated to 1536-dim)
  -> MemoryNetwork: soft attention over concept embeddings
  -> Autoencoder: trained on safe data, computes reconstruction error
  -> Decision: high error = Unsafe, low error = Safe
  -> Test-time adaptation: updates least-used concept embeddings with residuals
```

**Key files:**

- `run_paper_eval.py` — main eval script
- `memory_network.py` — core module: CLIP integration, soft attention, MLP classifier
- `concept.json` — 1300 unsafe concept embeddings across 14 harm categories

**Dataset loaders** (under `datasets/` package):
- `datasets/mmsafety.py` — MM-SafetyBench, `attack_success.json` only (not failure)
- `datasets/figstep.py` — FigStep
- `datasets/jailbreakv_nano.py` — JailbreakV-Nano
- `datasets/mmvet.py` — MM-Vet (safe/benign data)

**Archived** (not used by active scripts): `demo.ipynb`, `run_full.py`, `generate_dataloader.py`, `utils.py`, `ShieldEval.py`, `datasets/jailbreakv_28k.py`

## Known Issues

- `datasets/mmsafety.py` must use `attack_success.json` only. Including `attack_failure.json` (1369 samples of jailbreaks the VLM already refused) tanks MM-SafetyBench AUROC from ~0.90 to ~0.83.
- `transformers 5.x` returns `BaseModelOutputWithPooling` from `get_text/image_features()` instead of a tensor. `memory_network.py` handles this by extracting `.pooler_output`.
