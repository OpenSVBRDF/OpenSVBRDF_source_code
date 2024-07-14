from typing import Tuple
import torch
import torch.nn.functional as F
import math


def cosine_sample_hemisphere(
    size: Tuple
) -> torch.Tensor:

    u1 = torch.rand(*size, 1)
    u2 = torch.rand(*size, 1)

    # Uniformly sample disk.
    r = torch.sqrt(u1)
    phi = 2 * math.pi * u2
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)

    # Project up to hemisphere.
    z = torch.sqrt(1 - x * x - y * y)

    return torch.cat([x, y, z], dim=len(size))


def reflect(
    i_: torch.Tensor,
    n_: torch.Tensor,
) -> torch.Tensor:

    assert(len(i_.size()) == 2)
    assert(len(n_.size()) == 3)
    n = F.normalize(n_, dim=2)

    i = i_.unsqueeze(1)  
    dot = torch.sum(n * i, dim=2, keepdim=True)

    return 2 * dot * n - i
