# JailDAM Reproduction Report

**Paper:** arXiv:2504.03770 (COLM 2025)  
**Repo:** https://github.com/ShenzheZhu/JailDAM  
**Date:** 2026-04-09  
**GPU:** RTX 3090 24GB | PyTorch 2.4.1+cu124 | transformers 5.5.1

---

## Summary

Our results are consistent with the original repo's `demo.ipynb` output, but **both** produce inflated metrics compared to the paper. The demo is a simplified demonstration, not the paper's evaluation code.

### Comparison

| Run | Dataset | AUROC | AUPRC | F1 |
|-----|---------|-------|-------|----|
| **Original repo demo.ipynb** | MM-SafetyBench (1680) | 0.9988 | 0.9999 | 0.9940 |
| **Our run** | FigStep (500) | 0.9999 | 1.0000 | 0.9990 |
| **Our run** | JailbreakV-Nano (410) | 0.9943 | 0.9986 | 0.9783 |
| **Paper Table 2** | MM-SafetyBench | 0.9472 | 0.9155 | — |
| **Paper Table 2** | FigStep | 0.9608 | 0.9616 | — |
| **Paper Table 2** | JailBreakV-28K | 0.9465 | 0.9464 | — |
| **Paper Table 2** | Overall | 0.9550 | 0.9530 | — |

The demo.ipynb gives ~0.99 while the paper reports ~0.95. The demo evaluation protocol differs from the paper's — likely the paper uses a harder evaluation with all three unsafe datasets combined and a more rigorous train/test split.

---

## Raw terminal output

### FigStep (run_figstep.py, seed=42)

```
Device: cuda  |  Seed: 42

=== Loading datasets ===
FigStep unsafe : 500 samples
MM-Vet safe    : 517 samples

Reduced concept embeddings shape: torch.Size([800, 768])
Train safe: 413  |  Val safe: 104  |  OOD unsafe: 500 (ALL)

=== Training autoencoder (safe data only) ===
Epoch [1/5]  AE Loss: 13.3900  Concept Loss: 2.4544
Epoch [2/5]  AE Loss: 0.3358  Concept Loss: 2.3851
Epoch [3/5]  AE Loss: 0.2932  Concept Loss: 2.3326
Epoch [4/5]  AE Loss: 0.1393  Concept Loss: 2.2831
Epoch [5/5]  AE Loss: 0.0990  Concept Loss: 2.2338

=== Evaluating: val_safe + ALL FigStep ===
  Dataloader done in 22.0s
  Dataloader done in 100.0s

=== Results ===
Safe samples evaluated : 104
Unsafe samples (FigStep): 500
AUROC     : 0.9999
AUPR      : 1.0000
F1        : 0.9990
Precision : 0.9980
Recall    : 1.0000
Threshold : 160.0685
Avg latency: 13.39 ms/input
```

### JailbreakV-Nano (run_jailbreakv_nano.py, seed=42)

```
Device: cuda  |  Seed: 42

=== Loading datasets ===
JailbreakV-Nano unsafe : 410 samples
MM-Vet safe            : 517 samples

Reduced concept embeddings shape: torch.Size([800, 768])
Train safe: 413  |  Val safe: 104  |  OOD unsafe (JailbreakV-Nano): 410 (ALL)

=== Training autoencoder (safe data only) ===
Epoch [1/5]  AE Loss: 13.3900  Concept Loss: 2.4544
Epoch [2/5]  AE Loss: 0.3358  Concept Loss: 2.3851
Epoch [3/5]  AE Loss: 0.2932  Concept Loss: 2.3326
Epoch [4/5]  AE Loss: 0.1393  Concept Loss: 2.2831
Epoch [5/5]  AE Loss: 0.0990  Concept Loss: 2.2338

=== Evaluating: val_safe + ALL JailbreakV-Nano ===
  Dataloader done in 21.4s
  Dataloader done in 87.7s

=== Results ===
Safe samples evaluated          : 104
Unsafe samples (JailbreakV-Nano): 410
AUROC     : 0.9943
AUPR      : 0.9986
F1        : 0.9783
Precision : 0.9689
Recall    : 0.9878
Threshold : 105.0733
Avg latency: 13.74 ms/input
```

### Original repo demo.ipynb (MM-SafetyBench, from author's environment)

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

### Full run (run_full.py, seed=42) — MM-SafetyBench + FigStep + JailbreakV-Nano

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
| JailbreakV-Nano | 0.9943 | — (not in paper) |
| Overall | 0.9999 | 0.9550 |

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

## Environment fix

`transformers 5.5.1` returns `BaseModelOutputWithPooling` from `get_text/image_features()` instead of a tensor. Patched `generate_dataloader.py` and `memory_network.py` to extract `.pooler_output`.

## Files created

- `run_figstep.py` — FigStep evaluation (fixed seed, all 500 samples)
- `run_jailbreakv_nano.py` — JailbreakV-Nano evaluation (fixed seed, all 410 samples)
- `UnsafeVLMDataset_jailbreakv_nano.py` — Dataset loader for JailbreakV-Nano
