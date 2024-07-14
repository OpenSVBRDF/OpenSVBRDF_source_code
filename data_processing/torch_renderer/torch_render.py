import torch
import math
import numpy as np

import random
from typing import List
import trimesh
import torch.nn.functional as F
from onb import ONB
from render_utils import compute_form_factors_utils
from materials.ggx_brdf import GGX_BRDF
from setup_config import SetupConfig
from ray_trace import RayTrace


class TorchRender(object):
    """
    Generate lumitexel with pytorch.

    The point is with GGX brdf and illuminated by many point lights.

    Assume the intensity of light l is I(l), the normal of light is
    nl, the position of light is xl. The position of the point is xp,
    the normal of the point is np, the brdf of the point is fr. Then
    the lumi of the point can be calculated according to the formula
    below:

    B(I, p) = fr * (wi * np) * (-wi * nl) / ||xl - xp||^2 * I(l)

    wi, np, nl, xl, xp are all defined in world space.
    """

    scalar = 1e4 * math.pi * 1e-2

    def __init__(
        self,
        setup_config: SetupConfig,
        mesh: trimesh.Trimesh = None,
    ) -> None:
        self.setup = setup_config
        self.material = GGX_BRDF
        self.ray_trace = None if mesh is None else RayTrace(mesh)
    
    def set_mesh(
        self,
        mesh: trimesh.Trimesh
    ):
        self.ray_trace = RayTrace(mesh)

    def generate_lumitexel(
        self,
        input_params : torch.Tensor,
        position: torch.Tensor,
        global_custom_frame: List[torch.Tensor] = None,
        use_custom_frame: str = "",
        pd_ps_wanted: str = "both",
        specular_component: str = "D_F_G_B",
    ) -> torch.Tensor:
        """
        Args:
            input_params: a tensor of shape (batch, len). len=7/11.
                n_2d, theta, ax, ay, pd, ps
            position: position in world space, of shape (batch, 3)
            global_custom_frame: a list of n, t, b frame. the shape of 
                n, t, b is (batch, 3)
            use_custom_frame: "ntb" provide ntb frame to calculate rather
                than auto generation
            pd_ps_wanted: "both", "ps_only", "pd_only"
            specular_component: the ingredient of BRDF, usually "D_F_G_B", B means bottom

        Returns:
            lumitexel: tensor of shape (batch, lightnum, channel)
            end_points: n, n_dot_w
        """
        end_points = {}
        device = input_params.device

        cam_pos = self.setup.camera.get_cam_pos(device)
        cam_pos = cam_pos.unsqueeze(0).repeat(input_params.size(0), 1)

        lumi, end_points = self.generate_direct_lumi(
            input_params,
            position,
            cam_pos,
            global_custom_frame,
            use_custom_frame,
            pd_ps_wanted,
            specular_component
        )

        return lumi, end_points
    
    def generate_visibility(
        self,
        mesh_pos: torch.Tensor,
        downsample: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            mesh:
            mesh_pos: the points on mesh, (N, 3)
        """
        if self.ray_trace is None:
            print("Error, torch_render doesn't have a mesh.")
            return

        light_pos = self.setup.get_light_poses("cpu")

        if downsample:
            light_pos = torch.from_numpy(self.setup.get_downsampled_light_poses(light_pos.numpy()))

        ray_origins = mesh_pos
        ray_dirs = light_pos.unsqueeze(0) - mesh_pos.unsqueeze(1)

        hit = self.ray_trace.intersects_any(ray_origins, ray_dirs)
        if downsample:
            hit = self.setup.upsample_data(hit.T).T
        else:
            hit = hit.float().numpy()

        return torch.from_numpy(1 - hit)

    def generate_direct_lumi(
        self,
        input_params: torch.Tensor,
        point_pos: torch.Tensor,
        view_pos: torch.Tensor,
        global_custom_frame: List[torch.Tensor] = None,
        use_custom_frame: str = "",
        pd_ps_wanted: str = "both",
        specular_component: str = "D_F_G_B",
    ):
        """
        Args:
            input_params: shape (batch, len). len=7/11. n_2d, theta, ax, ay, pd, ps
            point_pos: shape (batch, 3), point position in world space
            view_pos: shape (batch, 3), view position in world space
            global_custom_frame: a list of n, t, b frame. the shape of 
                n, t, b is (batch, 3)
            use_custom_frame: "ntb" provide ntb frame to calculate rather
                than auto generation
            pd_ps_wanted: "both", "ps_only", "pd_only"
            specular_component: the ingredient of BRDF, usually "D_F_G_B", B means bottom

        Returns:
            lumitexel: tensor of shape (batch, lightnum, channel)
            meta: n, n_dot_w
        """
        meta = {}
        batch_size = input_params.size(0)
        device = input_params.device

        # split input parameters to position and others
        if input_params.size(1) == 7:
            n_2d, theta, ax, ay, pd, ps = torch.split(input_params,
                                                      [2, 1, 1, 1, 1, 1],
                                                      dim=1)
        elif input_params.size(1) == 11:
            n_2d, theta, ax, ay, pd, ps = torch.split(input_params,
                                                      [2, 1, 1, 1, 3, 3],
                                                      dim=1)
        else:
            print("error input param len: {}".format(input_params.size(1)))
            exit(-1)

        # get setup
        light_normals = self.setup.get_light_normals(device)
        light_poses = self.setup.get_light_poses(device)
        cam_pos = view_pos.to(device)
        light_num = light_normals.size(0)
        
        # build local frame.
        # the calculation of the lumitexel is in the local frame ntb, n can
        # be seen as the normal of the point
        view_dir = F.normalize(cam_pos - point_pos, dim=1)

        # build shading coordination
        local_onb = ONB(batch_size)

        if "n" in use_custom_frame:
            n = global_custom_frame[0]
            if "t" in use_custom_frame:
                t = global_custom_frame[1]

                b = global_custom_frame[2]

                local_onb.build_from_ntb(n,t,b)
                # print("Error, not support t!")
                # exit(-1)
            else:
                local_onb.build_from_w(n)
                local_onb.rotate_frame(theta)
                t = local_onb.u()
            
            
        else:
            # build geometric coordination
            frame_onb = ONB(batch_size)
            frame_onb.build_from_w(view_dir)
            # local_onb is based on geometric coordination
            local_onb.build_from_n2d(n_2d, theta)

            frame_t, frame_b, frame_n = frame_onb.u(), frame_onb.v(), frame_onb.w()
            local_t, local_b, local_n = local_onb.u(), local_onb.v(), local_onb.w()

            n = local_n[:, [0]] * frame_t + local_n[:, [1]] * frame_b + local_n[:, [2]] * frame_n
            t = local_t[:, [0]] * frame_t + local_t[:, [1]] * frame_b + local_t[:, [2]] * frame_n
            b = torch.cross(n, t)

            # convert local_onb from geometric coordination to global coordination
            local_onb.build_from_ntb(n, t, b)

        wi = light_poses.unsqueeze(0) - point_pos.unsqueeze(1)
        wi = F.normalize(wi, dim=2)

        # get normalized normal
        n = local_onb.w()
        t = local_onb.u()

        # compute lumi
        form_factors = compute_form_factors_utils(point_pos, n, light_poses,
                                            light_normals, True)

        pd_ps_code = torch.ones(n.size(0), light_num)
        if pd_ps_wanted == "pd_only":
            pd_ps_code[:, :] = 0
        elif pd_ps_wanted == "ps_only":
            pd_ps_code[:, :] = 1
        else:
            pd_ps_code = None
        lumi, meta = self.material.eval(local_onb, wi, view_dir, ax, ay, pd, ps,
                                        pd_ps_code, specular_component)

        lumi = lumi * form_factors * TorchRender.scalar

        # check special case
        wi_dot_n = torch.sum(wi * n.unsqueeze(1), dim=2, keepdim=True)
        lumi = torch.where(torch.lt(wi_dot_n, 1e-6),
                           torch.zeros_like(lumi),
                           lumi)
        n_dot_wo = torch.sum(view_dir * n, dim=1, keepdim=True)
        meta["n_dot_wo"] = n_dot_wo
        n_dot_wo = n_dot_wo.unsqueeze(1).repeat(1, light_num, 1)

        lumi = torch.where(torch.lt(n_dot_wo, 0.0),
                           torch.zeros_like(lumi),
                           lumi)
        
        if pd_ps_wanted == "both":
            diff_lumi = meta["pd"] * form_factors * TorchRender.scalar
            spec_lumi = meta["ps"] * form_factors * TorchRender.scalar
            diff_lumi = torch.where(torch.lt(wi_dot_n, 1e-6),
                           torch.zeros_like(diff_lumi),
                           diff_lumi)
            spec_lumi = torch.where(torch.lt(n_dot_wo, 0.0),
                            torch.zeros_like(spec_lumi),
                            spec_lumi)
            meta["diff_lumi"] = diff_lumi
            meta["spec_lumi"] = spec_lumi

        meta["n"] = n
        meta["t"] = t
        return lumi, meta

    def visualize_lumi(
        self,
        lumi: torch.Tensor
    ) -> torch.Tensor:
        batch_size = lumi.size(0)
        channel = lumi.size(2)
        device = lumi.device
        H, W = self.setup.get_vis_img_size()

        tmp_img = torch.zeros(batch_size, H, W, channel, dtype=lumi.dtype).to(lumi.device)

        tmp_img[:, self.setup.get_visualize_map(device)[:, 1], self.setup.get_visualize_map(device)[:, 0]] = lumi

        if channel == 1:
            tmp_img = tmp_img.repeat(1, 1, 1, 3)

        return tmp_img
    
    def sample_mask(self, light_poses, position):
        """
        Args:
            light_poses: a tensor of shape (1, lightnum, 3)
            position: a tensor of shape (batch, 3)
        
        ```
        -----------------------
        |                     ^
        |        mask         | mask_axis_y
        |                     |                     z
        |    mask_axis_x      |                     |
        <----------------------               x ____|
        ```
        Returns:
            sample: a tensor of shape (batch, lightnum, 2)
        """
        device = position.device

        anchor, offset1, offset2 = self.setup.get_mask_data(device)

        normal = F.normalize(torch.cross(offset1, offset2).unsqueeze(0)).squeeze(0).to(device)

        d = torch.dot(normal, anchor).float().to(device)
        v1 = offset1 / torch.dot(offset1, offset1)
        v2 = offset2 / torch.dot(offset2, offset2)

        position = position.unsqueeze(1).to(device)

        ray_dir = light_poses - position

        normal = normal.unsqueeze(0).unsqueeze(0)
        dt = torch.sum(ray_dir * normal, dim=2)
        d = d.unsqueeze(0)
        t = (d - torch.sum(normal * position, dim=2)) / dt

        p = position + ray_dir * t.unsqueeze(2)
        vi = p - anchor.unsqueeze(0).unsqueeze(0)

        a1 = torch.sum(v1 * vi, dim=2)
        a2 = torch.sum(v2 * vi, dim=2)

        sample = torch.stack((a1, a2), dim=2)

        return sample

    def multi_sample_mask(
        self,
        _light_poses: torch.Tensor,
        position: torch.Tensor,
        light_size: int = 2,
        num: int = 25,
    ) -> torch.Tensor:
        """
        Multisample the planar light source.

        Args:
            light_poses: the central position of the light, shape (1, lightnum, 3)
            position: 
            light_size: the size of the planar light
            num: sample times. For example 25, 49, 81
        
        Returns:
            samples: a tensor of shape (batch, lightnum, num, 2)
        """
        sqrt_num = int(math.sqrt(num))
        assert(sqrt_num * sqrt_num == num)
        assert(sqrt_num % 2 == 1)
        offset_x = torch.arange(- (sqrt_num // 2),  sqrt_num // 2 + 1, 1).repeat_interleave(sqrt_num, 0).unsqueeze(1)
        offset_y = torch.zeros_like(offset_x)
        offset_z = torch.arange(- (sqrt_num // 2),  sqrt_num // 2 + 1, 1).repeat(sqrt_num).unsqueeze(1)
        offset = torch.cat([offset_x, offset_y, offset_z], dim=1).to(_light_poses.device)

        offset = offset.unsqueeze(0).repeat(1, _light_poses.size(1), 1) / (sqrt_num - 1) * light_size

        light_poses = _light_poses.repeat_interleave(num, 1) + offset

        sample = self.sample_mask(light_poses, position)

        return sample

    def generate_indirect_lumi(
        self,
        point_pos: torch.Tensor,
        view_pos: torch.Tensor,
        uv: torch.Tensor,
        sample_num: int,
        get_params,
        depth: int = 1,
        max_depth: int = 999999,
        rr_begin_depth: int = 5,
        p_rr: float = 1/2,
    ) -> torch.Tensor:
        """
        Args:
            point_pos: (batch, 3)
            view_pos: (batch, 3)
            uv: (batch, 2), the uv of the point_pos on mesh.
            get_params: a function to get params from uv.
        Returns:
            indirect_lumi: (batch, lightnum, channel)
        """
        if self.ray_trace is None:
            print("Error, torch_render doesn't have a mesh.")
            return

        batch_size = point_pos.size(0)

        params = get_params(uv)
        channel = 1 if params.size(1) == 7 else 3

        if depth > max_depth:
            return torch.zeros(batch_size, self.setup.get_light_num(), channel).to(point_pos.device)

        if depth > rr_begin_depth:
            ksi = random.random()
            if ksi <= p_rr:
                return torch.zeros(batch_size, self.setup.get_light_num(), channel).to(point_pos.device)
        else:
            ksi = 1

        # split input parameters to position and others
        if params.size(1) == 7:
            n_2d, theta, ax, ay, pd, ps = torch.split(params, [2, 1, 1, 1, 1, 1], dim=1)
        elif params.size(1) == 11:
            n_2d, theta, ax, ay, pd, ps = torch.split(params, [2, 1, 1, 1, 3, 3], dim=1)
        else:
            print("error input param len: {}".format(params.size(1)))
            exit(-1)

        wo = F.normalize(view_pos - point_pos, dim=1)

        # build shading coordination
        local_onb = ONB(batch_size)
        frame_onb = ONB(batch_size)
        frame_onb.build_from_w(wo)
        local_onb.build_from_n2d(n_2d.detach(), theta.detach())
        frame_t, frame_b, frame_n = frame_onb.u(), frame_onb.v(), frame_onb.w()
        local_t, local_b, local_n = local_onb.u(), local_onb.v(), local_onb.w()
        n = local_n[:, [0]] * frame_t + local_n[:, [1]] * frame_b + local_n[:, [2]] * frame_n
        t = local_t[:, [0]] * frame_t + local_t[:, [1]] * frame_b + local_t[:, [2]] * frame_n
        b = torch.cross(n, t)
        local_onb.build_from_ntb(n, t, b)

        # sample wi
        wi, is_diff = self.material.sample(local_onb, wo, sample_num, ax.detach(), ay.detach())

        hit_point, hit_uv = self.ray_trace.intersects_location(point_pos, wi)

        hit_point_col = hit_point.view(-1, 3)
        hit_uv_col = hit_uv.view(-1, 2)
        view_pos_col = torch.repeat_interleave(point_pos.unsqueeze(1), repeats=sample_num, dim=1)
        view_pos_col = view_pos_col.view(-1, 3)

        invalid = torch.all(hit_uv_col == 0, dim=1)

        params = get_params(hit_uv_col)

        direct_lumi, meta = self.generate_direct_lumi(params, hit_point_col,
                                                    view_pos_col, pd_ps_wanted="both")
        indirect_lumi = self.generate_indirect_lumi(hit_point_col, view_pos_col, hit_uv_col,
                                                    sample_num, get_params, depth=depth+1, max_depth=max_depth)

        mesh_pos_col = hit_point_col + meta['n'] * 0.1
        visibility = self.generate_visibility(mesh_pos_col.detach().cpu(), True).unsqueeze(2)
        
        visibility = visibility.to(device=direct_lumi.device, dtype=direct_lumi.dtype)
        lumi = direct_lumi * visibility + indirect_lumi

        lumi[invalid] = 0
        lumi = lumi.view(batch_size, sample_num, lumi.size(1), -1)

        fr, _ = self.material.eval(local_onb, wi, wo, ax, ay, pd, ps, is_diff)
        dot = torch.sum(wi * n.unsqueeze(1), dim=2, keepdim=True)
        pdf = self.material.pdf(local_onb, wi, wo, ax.detach(), ay.detach(), is_diff).unsqueeze(2)

        fr = fr.unsqueeze(2)
        dot = dot.unsqueeze(2)
        pdf = pdf.unsqueeze(2)
        indirect_lumi = torch.sum(lumi * fr * dot / (pdf / 2 + 1e-6), dim=1) / sample_num / ksi

        return indirect_lumi
