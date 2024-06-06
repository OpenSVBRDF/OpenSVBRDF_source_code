import torch
import torch.nn as nn
import os.path as osp
from tqdm import tqdm
import math
import sys

from texture import Texture
from mlp_model import MLPModel

TORCH_RENDER_PATH = "../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
from torch_render import TorchRender
from utils import write_rgb_image, read_exr


class LatentMLP(nn.Module):
    '''
    LatentMLP defines a multilayer perceptron model.

    Methods: 
        pred_params:    Predicts parameters such as the lighting direction and roughness from the input UV coordinates.
        export_texture: Saves the final texture results as .exr and .png files.
        forward:        Defines forward propagation.
        loss:           Calculates the loss during training.

    '''
    def __init__(
        self,
        torch_render: TorchRender,
        device: str,
        **kwargs
    ) -> None:
        super().__init__()
        self.torch_render = torch_render
        self.device = device
        self.step = 0

        self.pos_texture_file = kwargs.pop("position_texture", None)
        self.resolution = kwargs.pop("texture_resolution", 512)
        self.ps_range = kwargs.pop("ps_range", 5)
        self.axay_range = kwargs.pop("axay_range", 0.497)
        self.lambda_axay = kwargs.pop("lambda_axay", 0.1)
        self.lambda_m = kwargs.pop("lambda_m", 0.01)

        if self.pos_texture_file is None:
            print("Error, position_texture must be given.")

        self.inner_dim = 24
        self.param_encoding = Texture(self.resolution, self.resolution, self.inner_dim)

        pos_tex = torch.from_numpy(read_exr(self.pos_texture_file))
        for i in range(pos_tex.size(2)):
            pos_tex[:, :, i] = torch.Tensor.flipud(pos_tex[:, :, i])
        self.pos_texture = Texture(*(pos_tex.shape))
        self.pos_texture.set_parameters(pos_tex, False)

        self.n2d_mlp = MLPModel([self.inner_dim, 128, 128, 128, 2], normalizaiton=None, output_activation="sigmoid")
        self.theta_mlp = MLPModel([self.inner_dim, 128, 128, 128, 1], normalizaiton=None, output_activation="sigmoid")
        self.diffuse_albedo_mlp = MLPModel([self.inner_dim, 128, 128, 128, 3], normalizaiton=None, output_activation="sigmoid")
        self.specular_albedo_mlp = MLPModel([self.inner_dim, 128, 128, 128, 3], normalizaiton=None, output_activation="sigmoid")
        self.roughness_mlp = MLPModel([self.inner_dim, 128, 128, 128, 2], normalizaiton=None, output_activation="sigmoid")

        self.albedo_scalar = nn.Parameter(torch.ones(1), requires_grad=False)

    def pred_params(self, uvs):
        neck = self.param_encoding(uvs)

        n2d = self.n2d_mlp(neck)
        theta = self.theta_mlp(neck) * torch.pi * 2
        roughness = self.roughness_mlp(neck) * self.axay_range + 0.006

        diffuse_albedo = self.diffuse_albedo_mlp(neck) * self.albedo_scalar
        specular_albedo = self.specular_albedo_mlp(neck) * self.ps_range * self.albedo_scalar

        input_params = torch.cat([n2d, theta, roughness, diffuse_albedo, specular_albedo], dim=1)
        return input_params
    
    def get_position(self, uvs):
        p = self.pos_texture(uvs.detach())
        return p

    def forward(
        self,
        uvs: torch.Tensor,
        scalar: float = 625,
    ):
        pred_params = self.pred_params(uvs)
        point_pos = self.get_position(uvs)

        _, end_points = self.torch_render.generate_lumitexel(
            pred_params,
            point_pos,
            pd_ps_wanted="both",
        )

        diffuse_lumi = end_points["diff_lumi"] * scalar
        specular_lumi = end_points["spec_lumi"] * scalar

        end_points['pred_params'] = pred_params
        end_points['point_pos'] = point_pos

        return diffuse_lumi, specular_lumi, end_points

    def export_texture(self, exr_dir=None, png_dir=None, resolution=1024):
        half_dx =  0.5 / resolution
        half_dy =  0.5 / resolution
        xs = torch.linspace(half_dx, 1-half_dx, resolution, device=self.device)
        ys = torch.linspace(half_dx, 1-half_dy, resolution, device=self.device)
        xv, yv = torch.meshgrid([1 - xs, ys], indexing="ij")

        xy = torch.stack((yv.flatten(), xv.flatten())).t()

        with torch.no_grad():
            batch_size = 1024
            pred_params, pred_pos, pred_normal, pred_tangent = [], [], [], []
            for i in tqdm(range(xy.size(0) // batch_size + 1)):
                start = i * batch_size
                end = min((i + 1) * batch_size, xy.size(0))
                if start == end:
                    continue
                xy_tmp = xy[start:end]
                _, _, end_points = self.forward(xy_tmp)
                pred_params.append(end_points["pred_params"])
                pred_pos.append(end_points["point_pos"])
                pred_normal.append(end_points["n"])
                pred_tangent.append(end_points['t'])

            pred_params = torch.cat(pred_params, dim=0).to(self.device)
            pred_pos = torch.cat(pred_pos, dim=0).to(self.device)
            pred_normal = torch.cat(pred_normal, dim=0).to(self.device)
            pred_tangent = torch.cat(pred_tangent, dim=0).to(self.device)

        # process ax, ay and tangent
        n, t = pred_normal, pred_tangent
        b = torch.cross(n, t)
        pred_tangent = torch.where(pred_params[:, [3]] > pred_params[:, [4]], pred_tangent, b)
        pred_params[:, [3, 4]] = torch.where(pred_params[:, [3]] > pred_params[:, [4]], pred_params[:, [3, 4]], pred_params[:, [4, 3]])

        pred_params[:, [2]] = torch.where(pred_params[:, [3]] > pred_params[:, [4]], pred_params[:, [2]], pred_params[:, [2]] + torch.pi)
        pred_params[:, [2]] = torch.where(pred_params[:, [2]] <= torch.pi * 2, pred_params[:, [2]], pred_params[:, [2]] - torch.pi * 2)

        params_texture = pred_params.view(resolution, resolution, -1).detach().cpu()
        theta_texture = params_texture[:, :, [2]].repeat(1, 1, 3)
        pd_texture = params_texture[:, :, 5:8] / self.albedo_scalar.detach().cpu()
        ps_texture = params_texture[:, :, 8:11] / self.albedo_scalar.detach().cpu()
        ax_ay_img = torch.zeros_like(params_texture[:, :, :3])
        ax_ay_img[:, :, :2] = params_texture[:, :, [3, 4]]
        ax_ay_texture = ax_ay_img

        n2d_texture = torch.zeros_like(params_texture[:, :, :3])
        n2d_texture[:, :, :2] = params_texture[:, :, [0, 1]]
        
        normal_texture = pred_normal.view(resolution, resolution, -1).detach().cpu()
        normal_texture = normal_texture * 0.5 + 0.5

        tangent_texture = pred_tangent.view(resolution, resolution, -1).detach().cpu()
        tangent_texture = tangent_texture * 0.5 + 0.5

        if exr_dir is not None:
            write_rgb_image(osp.join(exr_dir, "ax_ay_texture.exr"), ax_ay_texture.numpy())
            write_rgb_image(osp.join(exr_dir, "pd_texture.exr"), pd_texture.numpy())
            write_rgb_image(osp.join(exr_dir, "ps_texture.exr"), ps_texture.numpy())
            write_rgb_image(osp.join(exr_dir, "normal_texture.exr"), normal_texture.numpy())
            write_rgb_image(osp.join(exr_dir, "tangent_texture.exr"), tangent_texture.numpy())
            write_rgb_image(osp.join(exr_dir, "theta_texture.exr"), theta_texture.numpy())
            write_rgb_image(osp.join(exr_dir, "n2d_texture.exr"), n2d_texture.numpy())

        if png_dir is not None:
            write_rgb_image(osp.join(png_dir, "ax_ay.png"), ax_ay_texture.numpy() ** (1 / 2.2) * 255)
            write_rgb_image(osp.join(png_dir, "pd.png"), pd_texture.numpy() ** (1 / 2.2) * 255)
            write_rgb_image(osp.join(png_dir, "ps.png"), (ps_texture.numpy() / 10) ** (1 / 2.2) * 255)
            write_rgb_image(osp.join(png_dir, "normal.png"), normal_texture.numpy() * 255)
            write_rgb_image(osp.join(png_dir, "tangent.png"), tangent_texture.numpy() * 255)
            write_rgb_image(osp.join(png_dir, "theta.png"), theta_texture.numpy() / math.pi / 2 * 255)
            write_rgb_image(osp.join(png_dir, "n2d.png"), n2d_texture.numpy() * 255)


        textures = torch.stack([theta_texture / torch.pi / 2, ax_ay_texture, pd_texture,
                                ps_texture / 5, normal_texture, tangent_texture], dim=0)
        textures = textures.permute(0, 3, 1, 2)
        return textures

    def loss(
        self,
        uvs: torch.Tensor,
        lumi: torch.Tensor,
        lp: torch.Tensor = None,
        m: torch.Tensor = None,
    ):
        pred_diff_lumi, pred_spec_lumi, end_points = self.forward(uvs, 3e3 / math.pi)
        pred_lumi = pred_diff_lumi + pred_spec_lumi

        if self.step == 1:
            print(f"Gt Lumitexel: max_val {lumi.max()}, mean_val: {lumi.mean()}")
            print(f"Pred Lumitexel: max_val {pred_lumi.max()}, mean_val: {pred_lumi.detach().mean()}")
            s = pred_lumi.detach().mean() / lumi.mean()
            self.albedo_scalar.data[0] = s
            print(f"Set albedo_scalar to {s}")

            latent_radiance = lumi.unsqueeze(3) * lp.unsqueeze(0)
            latent_radiance = torch.sum(latent_radiance, dim=1).transpose(2, 1)
            loss_m = torch.nn.MSELoss()(latent_radiance * self.albedo_scalar, m * self.albedo_scalar)
            print(f"Latent measurements MSE: {loss_m.item()}")

        # Loss lumitexel
        loss_lumi = torch.nn.MSELoss()(pred_lumi, lumi * self.albedo_scalar)

        # Loss: ax, ay regularization
        ax = end_points['pred_params'][:, [3]]
        ay = end_points['pred_params'][:, [4]]
        loss_ax_ay = torch.mean(ax * ay)

        # Loss measurements
        radiance = pred_lumi.unsqueeze(3) * lp.unsqueeze(0)
        radiance = torch.sum(radiance, dim=1).transpose(2, 1) 
        loss_m = torch.nn.MSELoss()(radiance, m * self.albedo_scalar)

        loss = loss_lumi + loss_ax_ay * self.lambda_axay + loss_m * self.lambda_m

        meta = {
            'loss_lumi': loss_lumi.item(),
            'loss_ax_ay': loss_ax_ay.item(),
            'loss_m': loss_m.item(),
            'pred_lumi': pred_lumi.detach(),
        }
        return loss, meta
