import os
from ..logger import get_logger

logger = get_logger(__name__)



AVAILABLE_BACKENDS = ['numpy']
BACKEND = 'numpy'

try:
    import torch
    AVAILABLE_BACKENDS.append('pytorch')
except ImportError:
    logger.warning('PyTorch not available. CPU calculation only!')
    
# Get the backend variable from environment varibles
# Defaults to 'numpy' if not set
backend = os.environ.get('PYRGF_BACKEND', 'numpy')

# Change the backend for matrix computation.
# Default backend is 'numpy'.
# If the PyTorch module intalled, 'pytorch' allows gpu computation.


if not backend in AVAILABLE_BACKENDS:
    logger.error(f'backend {backend} unknown or not supported.')
    logger.error(f'backends supported: {AVAILABLE_BACKENDS}')
    raise ValueError(f'backend {backend} unknown or not supported.')
else:
    BACKEND = backend
    
logger.info(f'Using backend: {BACKEND}')
    
if BACKEND == 'numpy':
    from .numpy_backend import *
elif BACKEND == 'pytorch':
    from .pytorch_backend import *

# def use_backend(backend):
#     '''
#     Change the backend for matrix computation.
#     Default backend is 'numpy'.
#     If the PyTorch module intalled, 'pytorch' allows gpu computation.
#     '''
#     global BACKEND
#     if not backend in AVAILABLE_BACKENDS:
#         logger.error(f'backend {backend} unknown or not supported.')
#         logger.error(f'backends supported: {AVAILABLE_BACKENDS}')
#         raise ValueError(f'backend {backend} unknown or not supported.')
#     else:
#         BACKEND = backend