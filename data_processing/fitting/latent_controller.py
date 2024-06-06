import torch
import numpy as np
import os.path as osp
import sys

TORCH_RENDER_PATH = "../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)

from setup_config import SetupConfig

AUTOENCODER_PATH="../finetune/"
sys.path.append(AUTOENCODER_PATH)
from AUTO_planar_scanner_net_inference import PlanarScannerNet


class LatentController(object):
    '''
    LatentController includes methods for loading latent models, measurements, and latent data.

    Methods: 
        __init__:          Initializes the LatentController with the model file, setup configuration directory, and device.
        load_model:        Loads a trained model from the model file and updates the parameters to PlanarScannerNet.
        load_measurements: Loads the measurement data.
        load_latent_data:  Loads the latent data from the specified directory.
        pred:              Makes predictions based on the input latent data of color and shape.
        get_light_pattern: Retrieves the light pattern.

    '''
    def __init__(
        self,
        model_file: str,
        setup_config_dir: str,
        device: str,
    ) -> None:
        self.setup = SetupConfig(setup_config_dir)
        self.model = self.load_model(model_file, device)
        self.device = device
        self.tex_resolution = 1024
        self.model_root = osp.dirname(model_file)

    def load_model(
        self,
        model_file: str,
        device: str,
    ) -> PlanarScannerNet:
        """
        Load latent model
        """
        train_configs = {}
        train_configs["training_device"] = 0
        train_configs["lumitexel_length"] = 64*64*3
        train_configs["shape_latent_len"] = 48
        train_configs["color_latent_len"] = 8
        inference_net = PlanarScannerNet(train_configs)
        pretrained_dict = torch.load(model_file)
        print("loading trained model...")

        something_not_found = False
        model_dict = inference_net.state_dict()
        for k,_ in model_dict.items():
            if k not in pretrained_dict:
                print("not found:", k)
                something_not_found = True
        if something_not_found:
            exit()

        model_dict = inference_net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        inference_net.load_state_dict(model_dict)
        for p in inference_net.parameters():
            p.requires_grad=False
        inference_net.to(device)
        inference_net.eval()
        return inference_net
    
    def load_measurements(
        self,
        data_root: str,
    ) -> torch.Tensor:
        m_path = osp.join(data_root, f"line_measurements_{self.tex_resolution}.bin")
        if osp.exists(m_path):
            measurements = np.fromfile(m_path, np.float32).reshape(-1, 64, 3)
            measurements = torch.from_numpy(measurements)
        else:
            measurements = None
        return measurements

    def load_latent_data(
        self,
        data_root: str,
    ):  
        shape_latent_len = self.model.lumitexel_net.shape_latent_len
        color_latent_len = self.model.lumitexel_net.color_latent_len
        all_latent_len = shape_latent_len + 3 * color_latent_len
        
        pf_latent = open(osp.join(data_root, f"latent/pass3_latent_{self.tex_resolution}.bin"), "rb")
        pf_latent.seek(0, 2)
        texel_num = pf_latent.tell() // all_latent_len // 4
        pf_latent.seek(0, 0)
        print("texel num : ", texel_num)
        positions = np.fromfile(osp.join(data_root, f"texture_1024/positions.bin"), np.float32).reshape([-1, 3])
        positions = torch.from_numpy(positions)
        assert positions.size(0) == texel_num

        latent_code = np.fromfile(pf_latent, np.float32).reshape([texel_num, all_latent_len])
        latent_code = torch.from_numpy(latent_code)

        color_latent = latent_code[:, :3 * color_latent_len].reshape([texel_num, 3, color_latent_len])
        shape_latent = latent_code[:, 3 * color_latent_len:].reshape([texel_num, 1, shape_latent_len]).repeat(1, 3, 1)

        color_shape_latent = torch.cat([color_latent, shape_latent],dim=-1)

        self.color_shape_latent = color_shape_latent.to(self.device)
        self.positions = positions.to(self.device)
        return color_shape_latent, positions

    def pred(
        self,
        color_shape_latent
    ):
        batch_size = color_shape_latent.size(0)
        latent_len = color_shape_latent.size(2)
        tmp_latent = color_shape_latent.reshape([batch_size*3, latent_len])

        _, tmp_nn_lumi = self.model(tmp_latent, input_is_latent=True)
        tmp_nn_lumi = torch.max(torch.zeros_like(tmp_nn_lumi), tmp_nn_lumi)
        tmp_nn_lumi = tmp_nn_lumi.reshape([batch_size, 3, -1]).permute(0, 2, 1)
        return tmp_nn_lumi
    
    def get_light_pattern(self) -> torch.Tensor:
        path = f"{self.model_root}/opt_W_64.bin"
        lp = None
        if osp.exists(path):
            lp = torch.from_numpy(np.fromfile(path, np.float32).reshape(64, -1, 3))
            lp = lp.permute(1, 2, 0)
        return lp