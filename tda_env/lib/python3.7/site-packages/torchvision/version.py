__version__ = '0.4.0'
git_version = '6b959eef1fba106a070cd3f300bca98124da843d'
from torchvision import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
