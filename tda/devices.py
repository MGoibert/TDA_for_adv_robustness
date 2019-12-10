import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nb_cuda_devices = torch.cuda.device_count()

logger.info(f"Found {nb_cuda_devices} compatible with CUDA")

if nb_cuda_devices > 0:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
