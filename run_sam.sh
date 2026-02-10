#!/bin/bash
source /opt/miniconda3/bin/activate sam3_env
cd sam3
# Enable CPU fallback for PyTorch ops not yet implemented on MPS (Apple Silicon)
export PYTORCH_ENABLE_MPS_FALLBACK=1
python start_sam_server.py
