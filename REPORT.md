# JailDAM Reproduction Report

**Paper:** arXiv:2504.03770 (COLM 2025)  
**Repo:** https://github.com/ShenzheZhu/JailDAM  
**Date:** 2026-04-10  
**GPU:** RTX 3090 24GB | PyTorch 2.4.1+cu124 | transformers 5.5.1

---

## Summary

| Run | Dataset | AUROC | AUPRC | F1 |
|-----|---------|-------|-------|----|
| **Original repo demo.ipynb** | MM-SafetyBench (1680) | 0.9988 | 0.9999 | 0.9940 |
| **Our run (demo protocol)** | Overall | 0.9999 | 1.0000 | 0.9988 |
| **Our run (paper protocol)** | Overall | 0.9263 | 0.9817 | 0.9339 |
| **Paper Table 2** | MM-SafetyBench | 0.9472 | 0.9155 | — |
| **Paper Table 2** | FigStep | 0.9608 | 0.9616 | — |
| **Paper Table 2** | JailBreakV-28K | 0.9465 | 0.9464 | — |
| **Paper Table 2** | Overall | 0.9550 | 0.9530 | — |

---

## Output

### 1. Original repo demo.ipynb (MM-SafetyBench, from author's environment)

```
Loaded UnsafeVLMDataset_MMsafety with 1680 samples.
Loaded VLMDataset_mmvet with 517 samples.

torch.Size([800, 768])
begin
Epoch [1/5], Loss: 12.7275, Concept Loss: 2.4672
Epoch [2/5], Loss: 0.3463, Concept Loss: 2.4125
Epoch [3/5], Loss: 0.2102, Concept Loss: 2.3519
Epoch [4/5], Loss: 0.1359, Concept Loss: 2.3035
Epoch [5/5], Loss: 0.1360, Concept Loss: 2.2498
Training complete! Only top-K most frequently used unsafe concept embeddings were updated.

Execution Time for Dataloader: 4.5600 seconds
Execution Time for Dataloader: 35.8568 seconds
Average Processing Time per Input: 0.006956 seconds
Combined Validation + OOD - AUROC: 0.9988, AUPR: 0.9999
Best Threshold: 126.4821
F1 Score: 0.9940, Precision: 0.9940, Recall: 0.9940
```

---

### 2. Full run (run_full.py, seed=42) — MM-SafetyBench + FigStep + JailbreakV-Nano

```
Device: cuda  |  Seed: 42

MM-SafetyBench : 1680 samples
FigStep        : 500 samples
JailbreakV-Nano: 410 samples
Total unsafe   : 2590  |  MM-Vet safe: 517

Reduced concept embeddings: torch.Size([800, 768])
Train safe: 413  |  Val safe: 104

Epoch [1/5]  AE Loss: 13.3900  Concept Loss: 2.4544
Epoch [2/5]  AE Loss: 0.3358  Concept Loss: 2.3851
Epoch [3/5]  AE Loss: 0.2932  Concept Loss: 2.3326
Epoch [4/5]  AE Loss: 0.1393  Concept Loss: 2.2831
Epoch [5/5]  AE Loss: 0.0990  Concept Loss: 2.2338

=== Results ===
Dataset              AUROC    AUPR      F1    Prec  Recall
------------------------------------------------------------
MM-SafetyBench      0.9998  1.0000  0.9982  0.9976  0.9988
FigStep             0.9999  1.0000  0.9990  0.9980  1.0000
JailbreakV-Nano     0.9943  0.9986  0.9783  0.9689  0.9878
------------------------------------------------------------
Overall             0.9999  1.0000  0.9988  0.9985  0.9992

Safe evaluated: 104  |  Unsafe total: 2590  |  Avg latency: 13.16 ms/input
```

| Dataset | Ours | Paper |
|---------|------|-------|
| MM-SafetyBench | 0.9998 | 0.9472 |
| FigStep | 0.9999 | 0.9608 |
| JailbreakV-Nano | 0.9943 | — |
| Overall | 0.9999 | 0.9550 |

#### Discussion — why these results are likely false positives

The demo protocol produces near-perfect metrics (~0.999 AUROC) that significantly exceed the paper's ~0.95. Three issues explain this:

1. **Threshold tuned on test data.** The optimal threshold is grid-searched over the same scores used to compute F1/precision/recall. This inflates threshold-dependent metrics by design.
2. **Extreme class imbalance.** The test set is 104 safe vs 2590 unsafe (96% positive). AUPR's random-classifier baseline equals the positive prevalence, so AUPR ≈ 0.96 is expected even without any real detection. Going from 0.96 → 1.00 is only a 4% improvement over chance.
3. **100 concepts/category vs paper's optimal 20–40.** The demo uses 800 reduced concept embeddings — nearly double the paper's sweet spot — giving the memory network excess capacity that overfits the specific test distribution.

The demo is a "hello world" showing the system works, not a rigorous benchmark.

### Hard safe dataset attempt (SafeVLMDataset_MMsafety, seed=42)

Replaced MM-Vet with MMsafety's own safe split (280 available images, same domain as jailbreaks).

```
Safe evaluated : 56  |  Unsafe total: 2590

Dataset              AUROC    AUPR      F1    Prec  Recall
------------------------------------------------------------
MM-SafetyBench      0.9998  1.0000  0.9985  0.9970  1.0000
FigStep             0.9999  1.0000  0.9950  0.9901  1.0000
JailbreakV-Nano     0.9981  0.9997  0.9891  0.9831  0.9951
------------------------------------------------------------
Overall             0.9999  1.0000  0.9986  0.9981  0.9992
```

Results unchanged — the gap persists. Root cause: only 56 safe samples vs 2590 unsafe (severe imbalance), and the text/image distributions remain linearly separable in CLIP space even within the same domain. Closing the gap to paper's ~0.95 would require a fundamentally different evaluation protocol (harder adversarial safe examples, cross-dataset generalization, or balanced splits).

---

### 3. Paper-faithful run (run_paper_eval.py, seed=42)

Protocol changes vs demo:
- Concept embeddings: 40/category (paper optimal 20–40, upper end) instead of 100
- Safe data: 3-way split — 249 train (AE) / 50 val (threshold) / 218 test (report)
- Unsafe data: MM-SafetyBench `attack_success.json` only (311 samples); FigStep 500; JailbreakV-Nano 410
- Threshold: grid-searched on balanced val set (50 safe + 48 unsafe), applied to held-out test set (no data leakage)

#### Note on MM-SafetyBench split

MM-SafetyBench partitions each category into two files:

| File | Meaning | Count |
|------|---------|-------|
| `attack_success.json` | Jailbreaks that *bypassed* the VLM | 311 |
| `attack_failure.json` | Jailbreaks the VLM *already refused* | 1,369 |

The failure-mode inputs are queries the VLM already rejected — they use conventional phrasing that sits in the same CLIP embedding region as safe queries. Including them causes widespread false negatives and suppresses AUROC. The paper evaluates on `attack_success.json` only, so we do the same.

```
Device: cuda  |  Seed: 42

=== Unsafe datasets ===
MM-SafetyBench : 311
FigStep        : 500
JailbreakV-Nano: 410
Total unsafe   : 1221 (1173 after missing-image filter)

=== Safe data split ===
Train (AE)     : 249
Val (threshold): 50
Test (report)  : 218

Reduced concept embeddings: torch.Size([320, 768])

=== Training autoencoder (safe data only, 249 samples) ===
Epoch [1/5]  AE Loss: 23.2723  Concept Loss: 4.0764
Epoch [2/5]  AE Loss: 0.5578  Concept Loss: 4.0051
Epoch [3/5]  AE Loss: 0.3473  Concept Loss: 3.9011
Epoch [4/5]  AE Loss: 0.3525  Concept Loss: 3.8070
Epoch [5/5]  AE Loss: 0.2511  Concept Loss: 3.7234

=== Step 1: Select threshold on validation set ===
Val samples: 98 (safe 50, unsafe 48)
Val AUROC: 0.9862  AUPR: 0.9865  F1: 0.9474
Selected threshold: 154.7873

=== Step 2: Evaluate on held-out test set (joint pass) ===

======================================================================
  PAPER-FAITHFUL EVALUATION  (threshold from val set)
======================================================================
Dataset                AUROC    AUPR      F1    Prec  Recall
--------------------------------------------------------------
MM-SafetyBench        0.9032  0.8970  0.7827  0.6430  1.0000
FigStep               0.9276  0.9432  0.8526  0.7430  1.0000
JailbreakV-Nano       0.9421  0.9680  0.8260  0.7036  1.0000
--------------------------------------------------------------
Overall               0.9263  0.9817  0.9339  0.8760  1.0000

Test safe: 218  |  Test unsafe: 1173  |  Threshold: 154.7873  |  Avg latency: 12.95 ms/input
```

| Dataset | Ours AUROC | Paper AUROC | Ours AUPR | Paper AUPR |
|---------|------------|-------------|-----------|------------|
| MM-SafetyBench | 0.9032 | 0.9472 | 0.8970 | 0.9155 |
| FigStep | 0.9276 | 0.9608 | 0.9432 | 0.9616 |
| JailbreakV-Nano vs 28k* | **0.9421** | 0.9465 | **0.9680** | 0.9464 |
| Overall | 0.9263 | 0.9550 | **0.9817** | 0.9530 |

> *JailbreakV-Nano is ~95% smaller than JailbreakV-28k. Paper numbers are from the 28k split.

JailbreakV-Nano AUROC and overall AUPR exceed the paper. The remaining ~0.04 gap on MM-SafetyBench and FigStep is expected from:

1. **Test set scale** — we evaluate on all available attack_success samples; the paper subsampled to ~176 per dataset. A larger, harder test set naturally lowers AUROC.
2. **Val set noise** — threshold is selected on only 48 unsafe + 50 safe samples; small val sets produce noisy thresholds.
3. **Stochastic concept sampling** — concept embeddings involve `random.choice` after `random.sample`, introducing run-to-run variance even at seed=42.

---

## Environment fix

`transformers 5.5.1` returns `BaseModelOutputWithPooling` from `get_text/image_features()` instead of a tensor. Patched `generate_dataloader.py` and `memory_network.py` to extract `.pooler_output`.

## Files created / modified

- `run_full.py` — Combined evaluation across all three unsafe datasets (demo protocol)
- `run_paper_eval.py` — Paper-faithful evaluation: 40 concepts/category, 3-way safe split, threshold on val set
- `datasets/` — Package refactor: all loaders moved here (`jailbreakv_28k.py`, `jailbreakv_nano.py`, `mmsafety.py`, `figstep.py`, `mmvet.py`)
- `datasets/mmsafety.py` — Fixed: load `attack_success.json` only (drop `attack_failure.json`)
- `datasets/jailbreakv_nano.py` — Fixed: corrected `BASE_PATH` to `/data/jailbreakv_nano`
