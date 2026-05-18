# Job RAG Pipeline


## CUDA (Windows)

This project defaults to CPU-only PyTorch in `requirements.txt`. To use CUDA:

1. Install the NVIDIA driver and CUDA Toolkit that matches your GPU. Use the official NVIDIA installer for Windows.
2. Install the CUDA-enabled PyTorch build:

```powershell
pip uninstall -y torch
pip install torch==2.6.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Note: CUDA wheels are not currently available for Python 3.13 on the PyTorch index. If you are on 3.13, use Python 3.12 (or conda) to install the CUDA build.

To confirm CUDA is available:

```powershell
python - << 'PY'
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
PY
```

## Ingestion

Open the notebook for the end-to-end ingestion workflow:

- `notebooks/ingestion.ipynb`
