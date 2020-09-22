import torch
import os

from tda.tda_logging import get_logger

logger = get_logger("Devices")

nb_cuda_devices = torch.cuda.device_count()

logger.info(f"Found {nb_cuda_devices} devices compatible with CUDA")

if nb_cuda_devices > 0:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if os.environ.get("FORCE_CPU", "0") == "1":
    device = torch.device("cpu")

logger.info(f"Device is {device}")
