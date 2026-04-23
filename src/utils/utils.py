import random
import numpy as np
import torch

#Set a global seed to ensure reproducibility of experiments.
def set_seed(seed: int = 42) -> None:
    #Set seed for Python's built-in random module.
    random.seed(seed)
    #Set seed for NumPy operations.
    np.random.seed(seed)
    #Set seed for PyTorch (CPU).
    torch.manual_seed(seed)

    #If GPU is available, set seed for CUDA as well.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

"""
 Controls randomness for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and GPU)

    Parameters:
    - seed: integer value used to initialize random number generators
    """