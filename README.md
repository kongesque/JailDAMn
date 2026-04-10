# JailDAM: Jailbreak Detection with Adaptive Memory for Vision-Language Model

Reproduction of [arXiv:2504.03770](https://arxiv.org/abs/2504.03770) (COLM 2025). See `REPORT.md` for results.

## Requirements

- Python 3.11, PyTorch 2.4+, CUDA 12+
- GPU required (tested on RTX 3090)

```bash
pip install torch transformers scikit-learn Pillow numpy
```

## Data

Download datasets and place under `/data/`:

| Dataset | Path |
|---------|------|
| MM-SafetyBench | `/data/mmsafety/` |
| FigStep | `/data/fig_step/` |
| JailbreakV-Nano | `/data/jailbreakv_nano/` |
| MM-Vet | `/data/mm-vet/` |

## Run

```bash
python run_paper_eval.py
```

Prints per-dataset and overall AUROC / AUPR / F1 with a threshold selected on a held-out val set.

## Citation

```bibtex
@article{nian2025jaildam,
  title={JailDAM: Jailbreak Detection with Adaptive Memory for Vision-Language Model},
  author={Nian, Yi and Zhu, Shenzhe and Qin, Yuehan and Li, Li and Wang, Ziyi and Xiao, Chaowei and Zhao, Yue},
  journal={arXiv preprint arXiv:2504.03770},
  year={2025}
}
```
