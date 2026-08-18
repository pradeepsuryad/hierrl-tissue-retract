"""Shared runtime bits. The paper's runs are single-CPU-core; CUDA is used if present."""
import random
import numpy as np
import torch

GLOBAL_SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = GLOBAL_SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
