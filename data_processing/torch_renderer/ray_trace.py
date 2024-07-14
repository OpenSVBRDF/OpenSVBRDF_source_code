import trimesh
import torch
import torch.nn.functional as F
import numpy as np


class RayTrace(object):
    '''
    RayTrace encapsulates ray tracing operations for a given object in valid volume.

    functions: 
        intersects_any:      determine whether the given rays intersect with the object.

        intersects_location: get the position and the UV coordinates where the rays intersect with the object.

    '''
    def __init__(
        self,
        mesh: trimesh.Trimesh,
    ) -> None:
        """
        Args:
            mesh: an object in valid volume
        """
        self.mesh = mesh
    
    def intersects_any(
        self,
        ray_origins: torch.Tensor,
        ray_dirs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ray_origins: (batch, 3)
            ray_dirs: (batch, N, 3), each origin has N directions
        
        Returns:
            hit: (batch, N)
        """
        batch = ray_origins.size(0)
        N = ray_dirs.size(1)

        ray_dirs = F.normalize(ray_dirs, dim=2)

        ray_origins_scale = torch.repeat_interleave(ray_origins, N, dim=0)

        ray_ori = ray_origins_scale.cpu().numpy()
        ray_dir = ray_dirs.reshape(-1, 3).cpu().numpy()

        ray_ori = ray_ori + ray_dir * 0.01

        hit = self.mesh.ray.intersects_any(
            ray_origins=ray_ori,
            ray_directions=ray_dir
        )

        hit = torch.from_numpy(hit.reshape(batch, N)).to(ray_origins.device)
        return hit
    
    def intersects_location(
        self,
        ray_origins: torch.Tensor,
        ray_dirs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ray_origins: (batch, 3)
            ray_dirs: (batch, N, 3), each origin has N directions
        Returns:
            hit_point: (batch, N, 3), If not hit, the position is (0, 0, 0)
            hit_uv: (batch, N, 2), If not hit, the uv is (0, 0)
        """
        batch = ray_origins.size(0)
        N = ray_dirs.size(1)

        ray_dirs = F.normalize(ray_dirs, dim=2)

        ray_origins_scale = torch.repeat_interleave(ray_origins, N, dim=0)

        ray_ori = ray_origins_scale.cpu().numpy()
        ray_dir = F.normalize(ray_dirs, dim=2).view(-1, 3).cpu().numpy()

        ray_ori = ray_ori + ray_dir * 0.01

        locations, index_ray, index_tri = self.mesh.ray.intersects_location(
            ray_origins=ray_ori,
            ray_directions=ray_dir,
            multiple_hits=False,
        )

        # Get the barycentric coordinates of the points in their respective triangle
        barys = trimesh.triangles.points_to_barycentric(self.mesh.vertices[self.mesh.faces[index_tri]], locations, method='cramer')
        uvs = np.einsum('ij,ijk->ik', barys, self.mesh.visual.uv[self.mesh.faces[index_tri]])

        hit_point = np.zeros_like(ray_ori)
        hit_point[index_ray] = locations
        hit_point = torch.from_numpy(hit_point.reshape(batch, N, 3)).to(ray_origins.device).float()

        hit_uv = np.zeros((ray_ori.shape[0], 2))
        hit_uv[index_ray] = uvs
        hit_uv = torch.from_numpy(hit_uv.reshape(batch, N, 2)).to(ray_origins.device).float()
        return hit_point, hit_uv