'''
 This is the experimental code for paper ``Ma et al., OpenSVBRDF: A Database of Measured Spatially-Varying Reflectance, SIGGRAPH Asia 2023``.
 This script is suboptimal and experimental.
 There may be redundant lines and functionalities.

 Xiaohe Ma, 2024/02
'''


'''
 This script performs the fitting via differentiable rendering.

'''

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import argparse
from pathlib import Path
import torch.utils.data as data
import sys

from latent_controller import LatentController
from latent_solver import LatentModelSolver
from latent_dataset import LatentDataset
from latent_mlp import LatentMLP
from utils import setup_seed, setup_multiprocess


TORCH_RENDER_PATH = "../../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
from torch_render import TorchRender
from setup_config import SetupConfig

PROJECT_ROOT = Path(__file__).parent.absolute()

train_device = "cuda:0"


OBJECT_CONFIG = {
    "satin": {
        "axay_range": 0.8, "ps_range": 10, "lambda_axay": 0.01, "lambda_m": 0.05
    }, 
    "fabric": {
        "axay_range": 0.5, "ps_range": 10, "lambda_axay": 0.1, "lambda_m": 0.001
    },
    "leather": {
        "axay_range": 0.5, "ps_range": 10, "lambda_axay": 0.01, "lambda_m": 0.005
    },
    "paper": {
        "axay_range": 0.5, "ps_range": 20, "lambda_axay": 0.05, "lambda_m": 0.005
    },
    "wallpaper": {
        "axay_range": 0.5, "ps_range": 10, "lambda_axay": 0.1, "lambda_m": 0.005
    },
    "wood": {
        "axay_range": 1, "ps_range": 10, "lambda_axay": 0.01, "lambda_m": 0.005
    },
    "woven": {
        "axay_range": 1, "ps_range": 10, "lambda_axay": 0.01, "lambda_m": 0.005
    },
    "metal": {
        "axay_range": 0.8, "ps_range": 10, "lambda_axay": 0.05, "lambda_m": 0.005
    },
    "ceramic": {
        "axay_range": 0.5, "ps_range": 10, "lambda_axay": 0.1, "lambda_m": 0.005
    }
}



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_class",type=str)
    parser.add_argument("data_root",type=str)
    parser.add_argument("save_root",default="texture_maps/")

    parser.add_argument('--train_device', type=str, default="cuda:0")
    parser.add_argument('--iter', type=int, default=120100)
    parser.add_argument("--main_cam_id",type=int,default=0)
    parser.add_argument("--tex_resolution",type=int,default=1024)

    parser.add_argument('--config_dir',type=str,                   default="../../torch_renderer/wallet_of_torch_renderer/lightstage/")
    parser.add_argument('--model_file',type=str,                   default="../../model/latent_48_24_500000_2437925.pkl")

    args, _ = parser.parse_known_args()

    global train_device
    train_device = args.train_device
    return args


def main():
    setup_seed(20)
    setup_multiprocess()

    args = parse_args()
    os.makedirs(args.save_root, exist_ok=True)

    setup = SetupConfig(args.config_dir, low_res=False)
    torch_render = TorchRender(setup)

    latent_controller = LatentController(args.model_file,
                                         args.config_dir,
                                         train_device)
    batch_size = 2**7

    latent_dataset = LatentDataset(latent_controller, args.save_root,args.data_root,batch_size, train_device)
    latent_loader = data.DataLoader(latent_dataset, batch_size=None, num_workers=0)

    ggx_mlp_config = {
        "position_texture" : args.save_root+"pos_texture.exr",
        "texture_resolution" : args.tex_resolution,
        "ps_range" :OBJECT_CONFIG[args.sample_class]["ps_range"],
        "axay_range" : OBJECT_CONFIG[args.sample_class]["axay_range"],
        "lambda_axay" :OBJECT_CONFIG[args.sample_class]["lambda_axay"],
        "lambda_m" : OBJECT_CONFIG[args.sample_class]["lambda_axay"]
    }

    ggx_mlp = LatentMLP(torch_render, train_device, **ggx_mlp_config).to(train_device)

    training_args = {
        "lr": 1e-3,
        "num_iters": args.iter,
        "device": train_device
    }

    solver = LatentModelSolver(ggx_mlp, latent_loader, **training_args)
    solver.train(latent_controller, args.data_root, args.save_root)


if __name__ == "__main__":
    main()