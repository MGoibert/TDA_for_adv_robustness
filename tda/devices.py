import torch

from tda.logging import get_logger

logger = get_logger("Devices")

nb_cuda_devices = torch.cuda.device_count()

logger.info(f"Found {nb_cuda_devices} compatible with CUDA")

if nb_cuda_devices > 0:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
