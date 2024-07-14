import os
import cv2
from typing import Tuple, Dict
import torch
import numpy as np
from camera import Camera


class SetupConfig(object):
    """
    This class manages the setup of lightstage.

    The setup should be saved in `wallet_of_torch_renderer` folder, include many
    configs: `cam_pos`, `lights_data`, `lights_angular`, `lights_intensity`,
    `visualize_map`, `rgb_tensor` and so on.
    """

    def __init__(
        self,
        config_dir: str,
        mask: torch.Tensor = None,
        low_res: bool = False,
    ) -> None:
        """
        Args:
            config_dir: the directory of the config
            mask: a bool tensor to remove unwanted lights. The shape is 
                (lightnum, ), if true, the light will be kept, else
                the light will be removed.
            low_res: low camera resolution, usually for optix simulation.
        """
        self.config_dir = config_dir
        self.mask = mask

        self.camera = Camera(self.config_dir, low_res)

        self.light_poses, self.light_normals = self._load_light_data(
            self.config_dir + "lights.bin")

        self.lights_angular = self._load_lights_angular(
            self.config_dir + "lights_angular.npy")

        self.lights_intensity = self._load_lights_intensity(
            self.config_dir + "lights_intensity.npy")

        self.img_size, self.visualize_map = self._load_visualize_map(
            self.config_dir + "visualize_config_torch.bin")

        self.rgb_tensor = self._load_rgb_tensor(
            self.config_dir + "color_tensor.bin"
        )

        self.mask_data = self._load_mask_data(
            self.config_dir + "mask.npy"
        )

        self.trained_idx = self._load_trained_idx(
            self.config_dir + "trained_idx.txt"
        )

    def _load_light_data(
            self, config_file: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load light position and normal from config file.

        TODO: Attention! The normal of lights is reversed! You should generate new setup config.

        Args:
            config_file: The config file is a bin file, and has data of shape
                (2, lightnum, 3), dtype=float32. In the first dim, 0 is position
                and 1 is normal.
        
        Returns:
            light_poses: a ndarray of shape (lightnum, 3)
            light_normals: a ndarray of shape (lightnum, 3)
        """
        light_data = np.fromfile(config_file, np.float32).reshape([2, -1, 3])
        light_poses, light_normals = light_data[0], light_data[1]
        if self.mask is not None:
            light_poses = light_data[0, self.mask, :]
            light_normals = light_data[1, self.mask, :]
        return light_poses, light_normals

    def _load_lights_angular(self, config_file: str) -> torch.Tensor:
        """
        Load angular distribution parameters of light intensity.

        The distribution is fitted with two cubic polynomials, therefore each
        light is with 8 parameters. See `compute_light_distribution` function 
        for more detail.

        Args:
            config_file: a npy file.
        
        Returns:
            lights_angular: a ndarray of shape (1, lightnum, 8)
        """
        lights_angular = None
        if os.path.isfile(config_file):
            lights_angular = np.load(config_file)
        if self.mask is not None:
            lights_angular = lights_angular[0, self.mask, :]
        return lights_angular

    def _load_lights_intensity(self, config_file: str) -> torch.Tensor:
        """
        Load the relative intensity of lights. 
        
        Args:
            config_file: a npy file
        
        Returns:
            lights_intensity: a ndarray of shape (lightnum, 3).
        """
        lights_intensity = None
        if os.path.isfile(config_file):
            lights_intensity = np.load(config_file)
        if self.mask is not None:
            lights_intensity = lights_intensity[self.mask, :]
        return lights_intensity

    def _load_visualize_map(
            self, config_file: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load visualize mapping data from config file.

        The sequence of lights in `light_poses` data. for example: our
        light board has 48*64 lights. The lights sequence in `light_poses`
        starts from right bottom corner, from bottom to top, from left to
        right. Then the map idx sequence will be 63 47 63 46 63 45 ... 
        63 0 62 47 62 46 ... 0 0.

        Args:
            config_file: The config file is a bin file with dtype int32. The
                first 2 integers are image size (W, H), after are map idx.

        Returns:
            img_size: a ndarray of shape (2, ), represents H, W
            visualize_map: a ndarray of shape (lightnum, 2)

        """
        with open(config_file, "rb") as pf:
            img_size = np.fromfile(pf, np.int32, 2)
            visualize_map = np.fromfile(pf, np.int32).reshape([-1, 2])
        return img_size, visualize_map

    def _load_rgb_tensor(self, config_file):
        """
        Load rgb tensor from config file

        Args:
            config_file:
        
        Returns:
            rgb_tensor: a ndarray of shape (3, 3, 3), (light, object, camera)
        """
        rgb_tensor = np.zeros((3, 3, 3))
        for i in range(3):
            rgb_tensor[i][i][i] = 1.0

        if os.path.isfile(config_file):
            rgb_tensor = np.fromfile(config_file, np.float32).reshape([3, 3, 3])
        else:
            print("Note: color tensor file is not found, use default color tensor!")

        return rgb_tensor

    def _load_mask_data(self, config_file):
        """
        Load mask data from config file

        Args:
            config_file: a npy file
        
        Returns:
            A dict contians `mask_anchor`, `mask_axis_x`, `mask_axis_y`
        """
        mask_data = np.load(config_file, allow_pickle=True).item()
        return mask_data

    def _load_trained_idx(self, config_file):
        if os.path.exists(config_file):
            trained_idx = np.loadtxt(config_file)
        else:
            trained_idx = []
        return trained_idx

    def print_configs(self) -> None:
        """
        Print the configs
        """
        print("[SETUP CONFIG]")
        print("cam_pos:", self.cam_pos)

    def get_rgb_tensor(self, device : str) -> torch.Tensor:
        """
        Returns:
            rgb_tensor: a tensor of shape (3, 3, 3), (light, object, camera)
        """
        return torch.from_numpy(self.rgb_tensor).to(device)
    
    def get_cam_pos(self, device : str) -> torch.Tensor:
        """
        Returns:
            cam_pos: shape (3, )
        """
        return self.camera.get_cam_pos(device)
    
    def get_lights_intensity(self, device : str) -> torch.Tensor:
        """
        Returns:
            lights_intensity: a tensor of shape (lightnum, 3).
        """
        return torch.from_numpy(self.lights_intensity).to(device)

    def get_lights_angular(self, device : str) -> torch.Tensor:
        """
        Returns:
            lights_angular: a tensor of shape (1, lightnum, 8)
        """
        return torch.from_numpy(self.lights_angular).to(device)

    def get_light_normals(self, device : str) -> torch.Tensor:
        """
        Returns:
            light_normals: a tensor of shape (lightnum, 3)
        """
        return torch.from_numpy(self.light_normals).to(device)

    def get_light_poses(self, device : str) -> torch.Tensor:
        """
        Returns:
            light_poses: a tensor of shape (lightnum, 3)
        """
        return torch.from_numpy(self.light_poses).to(device)
    
    def get_light_idx(self, light_pos: np.array) -> int:
        """
        Get the nearest light

        Args:
            light_pos: ndarray, (1, 3)
        """
        assert(light_pos.shape[0] == 1 and light_pos.shape[1] == 3)
        dist = self.light_poses - light_pos
        light_idx = np.argmin(np.sum(dist * dist, axis=1))
        return light_idx

    def get_trained_idx(self) -> list:
        """
        Returns:
            trained_idx: a list of trained_idx
        """
        return self.trained_idx

    def get_vis_img_size(self) -> Tuple[int, int]:
        """
        Returns:
            H, W
        """
        return self.img_size[0], self.img_size[1]
    
    def get_visualize_map(self, device : str) -> torch.Tensor:
        """
        Returns:
            visualize_map: a tensor of shape (lightnum, 2)
        """
        return torch.from_numpy(self.visualize_map).long().to(device)

    def get_mask_data(self, device : str) -> Dict:
        """
        Returns:
            mask_anchor:
            mask_axis_x:
            mask_axis_y: 
        """
        anchor = torch.from_numpy(self.mask_data['mask_anchor']).float().to(device)
        offset1 = torch.from_numpy(self.mask_data['mask_axis_x']).float().to(device)
        offset2 = torch.from_numpy(self.mask_data['mask_axis_y']).float().to(device)
        return anchor, offset1, offset2
    
    def get_light_num(self) -> int:
        return self.light_normals.shape[0]
    
    @classmethod
    def convert_xy_to_idx(cls, x, y):
        """
        Args:
            x, y: left top is (0, 0), x is vertical
        
        Returns:
            idx: light index, right buttom is 0, right top is 47.
        """
        return 48 * (63 - y) + (47 - x)
    
    @classmethod
    def convert_idx_to_xy(cls, idx):
        x = 47 - idx % 48
        y = 63 - idx // 48
        return x, y
    
    @classmethod
    def get_downsampled_light_poses(cls, ori_light_poses):
        """
        Args:
            light_poses: (3072, 3)
        Returns:
            light_poses downsampled
        """
        light_poses = np.zeros((3072 // 16, 3))
        cnt = 0
        for row in range(48 // 4):
            for col in range(64 // 4):
                ori_idx = cls.convert_xy_to_idx(4*row+2, 4*col+2)
                light_poses[cnt] = ori_light_poses[ori_idx]
                cnt += 1
        return light_poses
    
    @classmethod
    def upsample_data(
        cls,
        light_data: np.array
    ):
        """
        Args:
            light_data: (192, *)
        """
        new_light_data = np.zeros((3072, light_data.shape[1]))
        for i in range(light_data.shape[0]):
            row = i // 16
            col = i % 16
            for x in range(4*row, 4*(row+1)):
                for y in range(4*col, 4*(col+1)):
                    idx = cls.convert_xy_to_idx(x, y)
                    new_light_data[idx] = light_data[i]
        return new_light_data