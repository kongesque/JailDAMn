# JailDAM: Jailbreak Detection with Adaptive Memory for Vision-Language Model

Reproduction of [arXiv:2504.03770](https://arxiv.org/abs/2504.03770) (COLM 2025). See `REPORT.md` for results.

## Requirements

- Python 3.11, PyTorch 2.4+, CUDA 12+
- GPU required (tested on RTX 3090)

```bash
pip install transformers scikit-learn Pillow numpy
```

If PyTorch is not already installed (e.g. not using a RunPod/NGC image):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Data

Download datasets and place under `/data/`:

| Dataset | Download | Path |
|---------|----------|------|
| MM-SafetyBench | [download](https://drive.google.com/file/d/1KuntCKnviJhKN4WlMOI4GdD-LtLNQHOt/view?usp=drive_link) | `/data/mmsafety/` |
| FigStep | [download](https://drive.google.com/file/d/1_hK1r8YOuuY-0smjRPWwiGtJ1xDRyJ7c/view?usp=drive_link) | `/data/fig_step/` |
| JailbreakV-Nano | [download](https://drive.google.com/file/d/1gYh_xrLPjhJTv-2OFKqpUDQEQ0AZdPX5/view?usp=drive_link) | `/data/jailbreakv_nano/` |
| MM-Vet | [download](https://drive.google.com/file/d/1VHBWdJak4BNYTlfP63BsoxqhQyEfVH0N/view?usp=drive_link) | `/data/mm-vet/` |

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
