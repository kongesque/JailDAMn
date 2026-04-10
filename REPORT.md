# JailDAM Reproduction Report

**Paper:** arXiv:2504.03770 (COLM 2025)  
**Repo:** https://github.com/ShenzheZhu/JailDAM  
**Date:** 2026-04-10  
**GPU:** RTX 3090 24GB | PyTorch 2.4.1+cu124 | transformers 5.5.1

---

## Summary

| Run | Dataset | AUROC | AUPRC | F1 |
|-----|---------|-------|-------|----|
| Original repo demo.ipynb | MM-SafetyBench (1680) | 0.9988 | 0.9999 | 0.9940 |
| Our run (demo protocol) | Overall | 0.9999 | 1.0000 | 0.9988 |
| Our run (paper protocol) | Overall | 0.9263 | 0.9817 | 0.9339 |
| Paper Table 2 | MM-SafetyBench | 0.9472 | 0.9155 | - |
| Paper Table 2 | FigStep | 0.9608 | 0.9616 | - |
| Paper Table 2 | JailBreakV-28K | 0.9465 | 0.9464 | - |
| Paper Table 2 | Overall | 0.9550 | 0.9530 | - |

---

## Output

### 1. Original demo.ipynb (from author's environment)

```
Loaded UnsafeVLMDataset_MMsafety with 1680 samples.
Loaded VLMDataset_mmvet with 517 samples.

Epoch [1/5], Loss: 12.7275, Concept Loss: 2.4672
Epoch [2/5], Loss: 0.3463, Concept Loss: 2.4125
Epoch [3/5], Loss: 0.2102, Concept Loss: 2.3519
Epoch [4/5], Loss: 0.1359, Concept Loss: 2.3035
Epoch [5/5], Loss: 0.1360, Concept Loss: 2.2498

Combined Validation + OOD - AUROC: 0.9988, AUPR: 0.9999
Best Threshold: 126.4821
F1 Score: 0.9940, Precision: 0.9940, Recall: 0.9940
```

---

### 2. Full run (run_full.py, seed=42)

```
MM-SafetyBench : 1680 samples
FigStep        : 500 samples
JailbreakV-Nano: 410 samples
Total unsafe   : 2590  |  MM-Vet safe: 517
Train safe: 413  |  Val safe: 104

Epoch [1/5]  AE Loss: 13.3900  Concept Loss: 2.4544
Epoch [2/5]  AE Loss: 0.3358  Concept Loss: 2.3851
Epoch [3/5]  AE Loss: 0.2932  Concept Loss: 2.3326
Epoch [4/5]  AE Loss: 0.1393  Concept Loss: 2.2831
Epoch [5/5]  AE Loss: 0.0990  Concept Loss: 2.2338

Dataset              AUROC    AUPR      F1    Prec  Recall
------------------------------------------------------------
MM-SafetyBench      0.9998  1.0000  0.9982  0.9976  0.9988
FigStep             0.9999  1.0000  0.9990  0.9980  1.0000
JailbreakV-Nano     0.9943  0.9986  0.9783  0.9689  0.9878
Overall             0.9999  1.0000  0.9988  0.9985  0.9992

Safe: 104  |  Unsafe: 2590  |  Latency: 13.16 ms/input
```

These numbers are inflated. Three reasons:

1. Threshold is picked on the same data used to report F1 (leakage).
2. 104 safe vs 2590 unsafe means AUPR's random baseline is already 0.96 — going to 1.00 is only a 4-point real gain.
3. 100 concepts/category is above the paper's optimal 20-40, overfitting to the test distribution.

---

### 3. Paper-faithful run (run_paper_eval.py, seed=42)

Changes from demo: 40 concepts/category, 3-way safe split (249 train / 50 val / 218 test), threshold picked on val set only, MM-SafetyBench `attack_success.json` only.

On MM-SafetyBench, the original loader included both `attack_success.json` (311 samples) and `attack_failure.json` (1,369 samples). The failure samples are queries the VLM already refused, so they look like safe inputs in CLIP space and tank AUROC. Using success-only matches what the paper evaluates.

```
MM-SafetyBench : 311
FigStep        : 500
JailbreakV-Nano: 410
Train (AE): 249  |  Val: 50  |  Test: 218

Epoch [1/5]  AE Loss: 23.2723  Concept Loss: 4.0764
Epoch [2/5]  AE Loss: 0.5578  Concept Loss: 4.0051
Epoch [3/5]  AE Loss: 0.3473  Concept Loss: 3.9011
Epoch [4/5]  AE Loss: 0.3525  Concept Loss: 3.8070
Epoch [5/5]  AE Loss: 0.2511  Concept Loss: 3.7234

Val AUROC: 0.9862  |  Threshold: 154.7873

Dataset              AUROC    AUPR      F1    Prec  Recall
--------------------------------------------------------------
MM-SafetyBench      0.9032  0.8970  0.7827  0.6430  1.0000
FigStep             0.9276  0.9432  0.8526  0.7430  1.0000
JailbreakV-Nano     0.9421  0.9680  0.8260  0.7036  1.0000
Overall             0.9263  0.9817  0.9339  0.8760  1.0000

Safe: 218  |  Unsafe: 1173  |  Latency: 12.95 ms/input
```

| Dataset | Ours AUROC | Paper AUROC | Ours AUPR | Paper AUPR |
|---------|------------|-------------|-----------|------------|
| MM-SafetyBench | 0.9032 | 0.9472 | 0.8970 | 0.9155 |
| FigStep | 0.9276 | 0.9608 | 0.9432 | 0.9616 |
| JailbreakV-Nano vs 28k* | 0.9421 | 0.9465 | 0.9680 | 0.9464 |
| Overall | 0.9263 | 0.9550 | 0.9817 | 0.9530 |

*JailbreakV-Nano is ~95% smaller than JailbreakV-28k. Paper numbers are from the 28k split.

The remaining gap on MM-SafetyBench and FigStep comes down to test set size (we use all available samples, the paper subsampled), a small val set for threshold selection (48 unsafe + 50 safe), and some randomness in concept sampling.

---

## Notes

- `transformers 5.5.1` returns `BaseModelOutputWithPooling` instead of a tensor from `get_text/image_features()`. Fixed in `memory_network.py` by extracting `.pooler_output`.
- Datasets moved into `datasets/` package. `mmsafety.py` fixed to use `attack_success.json` only.
