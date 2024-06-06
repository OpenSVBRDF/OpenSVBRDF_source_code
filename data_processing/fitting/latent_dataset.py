import torch
import torch.utils.data as data
from pathlib import Path
from utils import write_rgb_image
from latent_controller import LatentController


class LatentDataset(data.Dataset):
    '''
    LatentDataset builds a dataset from the model file for fitting.

    Methods: 
        __init__:      Initializes the dataset with the latent controller, save root, data root, batch size, and device.
        valid_test:    Removes invalid points from the dataset.
        gen_uv:        Generates UV coordinates for further texture mapping.
        __load_fitting_data: Loads the fitting data from the specified directories.
        __len__:       Returns the total number of valid texels in the dataset.
        __getitem__:   Retrieves a batch of data at the specified index.
        get_light_pattern: Retrieves the light pattern from the latent controller.

    '''
    def __init__(
        self,        
        latent_controller: LatentController,
        save_root: str,
        data_root: str,
        batch_size: int,
        device: str,
    ) -> None:
        super().__init__()
        self.save_root = Path(save_root)
        self.latent_data_root = Path(data_root+"latent/")
        self.data_root = Path(data_root)
        self.m_data_root = Path(data_root+"texture_1024/")
        self.latent_controller = latent_controller
        self.batch_size = batch_size
        self.device = device

        latent_code, point_pos, uvs = self.__load_fitting_data()
        self.latent_code = latent_code
        self.point_pos = point_pos
        self.uvs = uvs
        self.pixel_num = self.latent_code.size(0)
        self.measurements = self.latent_controller.load_measurements(str(self.m_data_root))

        self.valid_test()

    def valid_test(self):
        print(f"Total texel num: {self.latent_code.size(0)}")
        latent_sum = torch.sum(self.latent_code, dim=(1, 2))
        valid_mask = ~torch.isnan(latent_sum)
        self.latent_code = self.latent_code[valid_mask]
        self.point_pos = self.point_pos[valid_mask]
        self.uvs = self.uvs[valid_mask]
        self.measurements = self.measurements[valid_mask]
        self.pixel_num = self.latent_code.size(0)
        print(f"Valid texel num: {self.latent_code.size(0)}")

    def gen_uv(self, resolution=1024):
        half_dx =  0.5 / resolution
        half_dy =  0.5 / resolution
        xs = torch.linspace(half_dx, 1-half_dx, resolution, device=self.device)
        ys = torch.linspace(half_dx, 1-half_dy, resolution, device=self.device)
        xv, yv = torch.meshgrid([1 - xs, ys])
        xy = torch.stack((yv.flatten(), xv.flatten())).t()
        return xy

    def __load_fitting_data(self):
        latent_code, positions = self.latent_controller.load_latent_data(str(self.data_root))
        
        uv_map = self.gen_uv()

        # save position texture
        pos_texture = positions.view(1024, 1024, 3).cpu().numpy()
        path = str(self.save_root / "pos_texture.exr")
        write_rgb_image(path, pos_texture)
        print(f"Generate position texture: {path}")

        return latent_code, positions, uv_map

    def __len__(self) -> int:
        return self.pixel_num
    
    def __getitem__(self, index: int):
        
        batch = torch.rand((self.batch_size, ), device=self.device, dtype=torch.float32)
        batch = (batch * self.pixel_num).long().cpu()
        latent_code = self.latent_code[batch]
        point_pos = self.point_pos[batch]
        uv = self.uvs[batch]

        m = self.measurements[batch] if self.measurements is not None else None
        lumitexel = self.latent_controller.pred(latent_code.to(self.device))

        train_data = {
            'lumitexel': lumitexel,
            'position': point_pos,
            'uv': uv,
            'measurements': m
        }
        return train_data

    def get_light_pattern(self):
        return self.latent_controller.get_light_pattern()