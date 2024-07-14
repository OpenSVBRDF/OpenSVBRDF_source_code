from typing import Tuple
import cv2
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F


class Camera(object):

    def __init__(
        self,
        config_dir: str,
        low_res: bool = False
    ) -> None:
        """
        Args:
            config_dir: config directory which contains cam_pos.bin, intrinsic.yml, extrinsic.yml
            low_res: low camera resolution. If `True`, the image size will be divided by 4. It's
                usually for optix simulation.
        """
        self.cam_pos = self._load_cam_pos(osp.join(config_dir, "cam_pos.bin"))
        self.intrinsic, self.image_size = self._load_intrinsic(osp.join(config_dir, "intrinsic.yml"), low_res)
        self.extrinsic = self._load_extrinsic(osp.join(config_dir, "extrinsic.yml"))

    def project_points(
        self,
        position: torch.Tensor,
        image_coordinates: bool,
    ) -> torch.Tensor:
        """
        Project the world position to camera coordinates / image 
        coordinates.

        Args:
            position: a tensor of shape (batch, 3)
            image_coordinates: If `True`, unprojects the points 
                to image coordinates. If `False` unprojects to 
                the camera view coordinates.

        Returns:
            image_coord: a tensor of shape (batch, 2)
        """
        device = position.device

        world_points = F.pad(position, (0, 1), value=1).to(device).T

        # world space to camera space
        camera_points = torch.mm(self.get_extrinsic(device), world_points)

        if not image_coordinates:
            return camera_points.T[..., :-1]

        # camera space to image space
        img_points = torch.mm(self.get_intrinsic(device), camera_points[:3, :])

        img_points = img_points.transpose(0, 1)             # (batch, 3)
        img_points = img_points[:, :2] / img_points[:, 2:3]
        img_points = img_points / self.get_img_size(device)

        return img_points
    
    def unproject_points(
        self,
        xy_depth: torch.Tensor,
        world_coordinates: bool
    ) -> torch.Tensor:
        """
        Transform input points from camera coodinates (screen)
        to the world / camera coordinates.

        Each of the input points `xy_depth` of shape (batch, 3) is
        a concatenation of the x, y location and its depth.

        For instance, for an input 2D tensor of shape `(num_points, 3)`
        `xy_depth` takes the following form:
            `xy_depth[i] = [x[i], y[i], depth[i]]`,
        for a each point at an index `i`.

        Args:
            xy_depth: torch tensor of shape (batch, 3).
            world_coordinates: If `True`, unprojects the points back to world
                coordinates using the camera extrinsics `R` and `T`.
                `False` ignores `R` and `T` and unprojects to
                the camera view coordinates.

        Returns
            new_points: unprojected points with the same shape as `xy_depth`.
        """
        device = xy_depth.device

        img_points = torch.cat([xy_depth[:, :2] * xy_depth[:, [2]], xy_depth[:, [2]]], dim=1)

        # image space to camera space
        camera_points = torch.mm(torch.inverse(self.get_intrinsic(device)), img_points.T)
        
        if not world_coordinates:
            return camera_points
        
        camera_points = F.pad(camera_points.T, (0, 1), value=1).T
        world_points = torch.mm(torch.inverse(self.get_extrinsic(device)), camera_points)
        return world_points.T[..., :-1]

    def _load_cam_pos(self, config_file: str) -> np.array:
        """
        Load camera position from config file.

        Args:
            config_file: The config file is a bin file, and data of shape
                 (3, ), dtype=float32
        
        Returns:
            cam_pos: shape (3, )
        """
        cam_pos = np.fromfile(config_file, np.float32)
        return cam_pos

    def _load_intrinsic(
        self,
        config_file: str,
        low_res: bool = False
    ) -> Tuple[np.array, np.array]:
        """
        Load camera intrinsic

        Args:
            config_file: opencv yml file
        
        Returns:
            intrinsic:  (3, 3)
            image_size: W, H
        """
        fs2 = cv2.FileStorage(config_file, cv2.FileStorage_READ)
        intrinsic = fs2.getNode('camera_matrix').mat()
        H = fs2.getNode('image_height').real()
        W = fs2.getNode('image_width').real()

        if low_res:
            intrinsic = intrinsic / 4
            intrinsic[2, 2] = 1
            H = H // 4
            W = W // 4

        return intrinsic, np.array([W, H])

    def _load_extrinsic(self, config_file: str) -> np.array:
        """
        Load camera extrinsic

        Args:
            config_file: opencv yml file
        
        Returns:
            extrinsic:  (4, 4)
        """
        fs2 = cv2.FileStorage(config_file, cv2.FileStorage_READ)
        extrinsic = np.zeros((4, 4))
        extrinsic[:3, :3] = fs2.getNode('rmat').mat()
        extrinsic[:3, 3] = fs2.getNode('tvec').mat()[:, 0]
        extrinsic[3, 3] = 1
        return extrinsic

    def get_intrinsic(self, device : str) -> torch.Tensor:
        """
        Returns:
            intrinsic: a tensor of shape (3, 3)
        """
        intrinsic = torch.from_numpy(self.intrinsic).float().to(device)
        return intrinsic
    
    def get_extrinsic(self, device : str) -> torch.Tensor:
        """
        Returns:
            intrinsic: a tensor of shape (4, 4)
        """
        extrinsic = torch.from_numpy(self.extrinsic).float().to(device)
        return extrinsic
    
    def get_img_size(self, device : str) -> torch.Tensor:
        """
        Returns:
            image_size: a tensor of shape (2, )
        """
        return torch.from_numpy(self.image_size).to(device)

    def get_cam_pos(self, device : str) -> torch.Tensor:
        """
        Returns:
            cam_pos: shape (3, )
        """
        return torch.from_numpy(self.cam_pos).to(device)
    
    def xy_to_raydir(self, xy_grid: torch.Tensor) -> torch.Tensor:
        """
        Convert the `xy_grid` input of shape `(batch, 2)` to raydir.

        Args:
            xy_grid: torch.tensor grid of image xy coords.

        Returns:
            raydir: raydir of each image point in world space.
                torch.tensor of shape (batch, 3)
        """
        device = xy_grid.device
        xy_depth = F.pad(xy_grid, (0, 1), value=1)

        world_pts = self.unproject_points(xy_depth, True)
        ray_dir = F.normalize(world_pts - self.get_cam_pos(device), dim=1)
        return ray_dir