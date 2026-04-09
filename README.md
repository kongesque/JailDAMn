# JailDAM: Jailbreak Detection with Adaptive Memory for Vision-Language Model

## System Requirements

- **OS**: Linux (x86-64)
- **Python**: 3.11
- **CUDA**: 12.1
- **GPU**: NVIDIA GPU with CUDA support required
- **Package manager**: Conda

## Quick Start

### 1. Download Data

Download the dataset from [Google Drive](https://drive.google.com/drive/folders/16Ge5BKIbj6bbD0fJOPYT9i8o5lr7HNgc) and place it in `./data`.

### 2. Set Up Environment

```bash
conda env create -f environment.yml
conda activate llava
```

### 3. Run

Open and execute `demo.ipynb` in Jupyter:

```bash
jupyter notebook demo.ipynb
```

## Citation

```bibtex
@article{nian2025jaildam,
  title={JailDAM: Jailbreak Detection with Adaptive Memory for Vision-Language Model},
  author={Nian, Yi and Zhu, Shenzhe and Qin, Yuehan and Li, Li and Wang, Ziyi and Xiao, Chaowei and Zhao, Yue},
  journal={arXiv preprint arXiv:2504.03770},
  year={2025}
}
```
