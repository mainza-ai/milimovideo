import sys
import torch

try:
    import ltx_core
    print("ltx_core imported successfully")
except ImportError as e:
    print(f"ltx_core import failed: {e}")

try:
    import ltx_pipelines
    print("ltx_pipelines imported successfully")
except ImportError as e:
    print(f"ltx_pipelines import failed: {e}")

print(f"Torch version: {torch.__version__}")
device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
    print("MPS (Metal Performance Shaders) is available and enabled!")
elif torch.cuda.is_available():
    device = "cuda"
    print("CUDA is available!")
else:
    print("Using CPU.")

# Test code paths that use device
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
# We can't easily test builder without a config/path, but validation of import is good enough.

print("Setup verification complete.")
