"""Reproductibilité : fixer toutes les seeds en un appel."""
from __future__ import annotations
import os, random
import numpy as np
import torch


def set_global_seed(seed: int, torch_deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    torch.backends.cudnn.benchmark = not torch_deterministic
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
