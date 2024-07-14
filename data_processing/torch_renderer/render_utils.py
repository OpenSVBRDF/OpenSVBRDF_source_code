from typing import Optional
import torch
import torch.nn.functional as F


def compute_form_factors_utils(position: torch.Tensor,
                         n: torch.Tensor,
                         light_poses: torch.Tensor,
                         light_normals: torch.Tensor,
                         with_cos: Optional[bool] = True) -> torch.Tensor:
    """
    Compute form factors in world space
    Formula: (wi * np) * (-wi * nl) / ||xl - xp||^2

    Args:
        position: the position of the point, of shape (batch, 3)
        n: the normal of the shading point, of shape (batch, 3)
        light_poses: shape (lightnum, 3)
        light_normals: shape (lightnum, 3)
        with_cos: if true, form factor add cos(ldir · light_normals)

    Returns:
        form_factor: (batch, lightnum, 1)
    """
    ldir = torch.unsqueeze(light_poses, dim=0) - torch.unsqueeze(position,
                                                                 dim=1)
    dist = torch.sqrt(torch.sum(ldir**2, dim=2,
                                keepdim=True))  
    ldir = F.normalize(ldir, dim=2)  

    a = torch.sum(ldir * torch.unsqueeze(n, dim=1), dim=2, keepdim=True)
    a = torch.clamp(a, min=0)  

    if not with_cos:
        return a

    b = dist * dist 
    
    c = torch.sum(ldir * torch.unsqueeze(light_normals, dim=0),
                  dim=2,
                  keepdim=True)
    c = torch.clamp(c, min=0)  

    return a / (b + 1e-6) * c
