import math
from threading import local
import torch
import torch.nn.functional as F
from materials.utils import cosine_sample_hemisphere
from materials.utils import reflect
from onb import ONB


class GGX_BRDF(object):

    @classmethod
    def sample(
        cls,
        local_onb: ONB,
        wo: torch.Tensor,
        sample_num: int,
        ax: torch.Tensor,
        ay: torch.Tensor,
    ):
        """
        Args:
            local_onb:
            wo: outgoing light dir in world space, shape (batch, 3)
            sample_num: 
            ax: shape (batch, 1)
            ay: shape (batch, 1)

        Returns:
            wi: sampled direction in world space, shape (batch, N, 3)
            is_diff: (batch, N), if == 0, sample pd, else sample ps
        """
        batch = wo.size(0)
        N = sample_num
        device = ax.device

        normal = local_onb.w()

        # decide to sample base on pd or ps
        is_diff = torch.rand(batch, N).to(device)
        is_diff[is_diff <= 0.5] = 0
        is_diff[is_diff > 0.5] = 1
        
        # sample pd
        unit_wi = cosine_sample_hemisphere((batch, N)).to(device)
        onb = ONB(batch)
        onb.build_from_w(normal)
        pd_wi = onb.inverse_transform(unit_wi)          # wi in world space

        # sample ps
        z1 = torch.rand(batch, N).to(device)
        z2 = torch.rand(batch, N).to(device)
        x = ax * torch.sqrt(z1) / torch.sqrt(1 - z1) * torch.cos(2 * math.pi * z2)
        y = ay * torch.sqrt(z1) / torch.sqrt(1 - z1) * torch.sin(2 * math.pi * z2)
        z = torch.ones(batch, N).to(device)
        local_vhalf = F.normalize(torch.stack([-x, -y, z], dim=2), dim=2)
        world_half = local_onb.inverse_transform(local_vhalf)
        ps_wi = reflect(wo, world_half)

        is_diff_scale = is_diff.unsqueeze(2).repeat(1, 1, 3)
        wi = torch.where(is_diff_scale == 0, pd_wi, ps_wi)
        return wi, is_diff

    @classmethod
    def pdf(
        cls,
        local_onb: ONB,
        wi: torch.Tensor,
        wo: torch.Tensor,
        ax: torch.Tensor,
        ay: torch.Tensor,
        is_diff: torch.Tensor,
    ):
        """
        Args:
            local_onb:
            wi: (batch, N, 3)
            wo: (batch, 3)
            ax: shape (batch, 1)
            ay: shape (batch, 1)
            is_diff: (batch, N)
        
        Returns:
            pdf: (batch, N)
        """
        N = wi.size(1)

        # pd
        normalized_wi = F.normalize(wi, dim=2)
        local_wi = local_onb.transform(normalized_wi)
        pdf_pd = local_wi[:, :, 2] / math.pi

        # ps
        local_wi = F.normalize(local_onb.transform(wi), dim=2)
        local_wo = F.normalize(local_onb.transform(wo), dim=1)

        local_half = F.normalize(local_wi + local_wo.unsqueeze(1).repeat(1, N, 1), dim=2)
        vhalf = local_half.clone()
        vhalf[:, :, 0] = vhalf[:, :, 0] / ax
        vhalf[:, :, 1] = vhalf[:, :, 1] / ay

        len2 = torch.sum(vhalf * vhalf, dim=2)
        D = 1.0 / (math.pi * ax * ay * len2 * len2)
        pdf_ps = D * local_half[:, :, 2] / 4 / torch.sum(local_half * local_wi, dim=2)

        pdf_ps = torch.where(local_wi[:, :, 2] > 0, pdf_ps, torch.zeros_like(pdf_ps).to(pdf_ps.device))

        pdf = torch.where(is_diff == 0, pdf_pd, pdf_ps)
        return pdf

    @classmethod
    def eval(
        cls,
        local_onb: ONB,
        wi: torch.Tensor,
        wo: torch.Tensor,
        ax: torch.Tensor,
        ay: torch.Tensor,
        pd: torch.Tensor,
        ps: torch.Tensor,
        is_diff: torch.Tensor = None,
        specular_component: str = "D_F_G_B",
    ) -> torch.Tensor:
        """
        Evaluate brdf in shading coordinate(ntb space).

        For each shading point, we will build a coordinate system to calculate
        the brdf. The coordinate is usually expressed as ntb. n is the normal
        of the shading point.

        Args:
            local_onb: a coordinate system to do shade
            wi: incident light in world space, of shape (batch, lightnum, 3)
            wo: outgoing light in world space, of shape (batch, 3)
            ax: shape (batch, 1)
            ay: shape (batch, 1)
            pd: shape (batch, channel), range [0, 1]
            ps: shape (batch, channel), range [0, 10]
            is_diff: shape (batch, lightnum), if the value is 0, eval "pd_only", else if
                the value is 1, eval "ps_only". If `is_diff` is None, means "both"
            specular_component: the ingredient of BRDF, usually "D_F_G_B", B means bottom

        Returns:
            brdf: (batch, lightnum, channel)
            meta: 
        """
        N = wi.size(1)

        # transform wi and wo to local frame
        wi_local = local_onb.transform(wi)
        wo_local = local_onb.transform(wo)

        meta = {}

        a = torch.unsqueeze(pd / math.pi, dim=1)
        b = cls.ggx_brdf_aniso(wi_local, wo_local, ax, ay, specular_component)
        ps = torch.unsqueeze(ps, dim=1)

        if is_diff is None:
            brdf = a + b * ps
        else:
            is_diff_ = is_diff.unsqueeze(2)
            brdf = a.repeat(1, N, 1) * (1 - is_diff_) + b * ps * is_diff_

        meta['pd'] = a
        meta['ps'] = b * ps

        return brdf, meta

    @classmethod
    def ggx_brdf_aniso(
        cls,
        wi: torch.Tensor,
        wo: torch.Tensor,
        ax: torch.Tensor,
        ay: torch.Tensor,
        specular_component: str
    ) -> torch.Tensor:
        """
        Calculate anisotropy ggx brdf in shading coordinate.
        
        Args:
            wi: incident light in ntb space, of shape (batch, lightnum, 3)
            wo: emergent light in ntb space, of shape (batch, 3)
            ax: shape (batch, 1)
            ay: shape (batch, 1)
            specular_component: the ingredient of BRDF, usually "D_F_G_B"
        
        Returns:
            brdf: shape (batch, lightnum, 1)
        """

        lightnum = wi.size(1)
        wo = torch.unsqueeze(wo, dim=1).repeat(1, lightnum, 1)

        wi_z = wi[:, :, [2]]
        wo_z = wo[:, :, [2]]
        denom = 4 * wi_z * wo_z  
        vhalf = F.normalize(wi + wo, dim=2)

        tmp = torch.clamp(1.0 - torch.sum(wi * vhalf, dim=2, keepdim=True), 0, 1)
        F0 = 0.04
        Fresnel = F0 + (1 - F0) * tmp * tmp * tmp * tmp * tmp

        axayaz = torch.unsqueeze(torch.cat([ax, ay, torch.ones_like(ax)], dim=1),
                                dim=1)
        vhalf = vhalf / (axayaz + 1e-6) 
        vhalf_norm = torch.norm(vhalf, dim=2, keepdim=True)
        length = vhalf_norm * vhalf_norm 
        D = 1.0 / (math.pi * torch.unsqueeze(ax, dim=1) *
                torch.unsqueeze(ay, dim=1) * length * length)

        G = cls.ggx_G1_aniso(wi, ax, ay, wi_z) * cls.ggx_G1_aniso(wo, ax, ay, wo_z)

        tmp = torch.ones_like(denom)
        if "D" in specular_component:
            tmp = tmp * D
        if "F" in specular_component:
            tmp = tmp * Fresnel
        if "G" in specular_component:
            tmp = tmp * G
        if "B" in specular_component:
            tmp = tmp / (denom + 1e-6)

        # some samples' wi_z/wo_z may less or equal than 0, should be 
        # set to zero. Maybe this step is not necessary, because G is
        # already zero.
        tmp_zeros = torch.zeros_like(tmp)
        static_zero = torch.zeros(1, device=wi.device, dtype=torch.float32)
        res = torch.where(torch.le(wi_z, static_zero), tmp_zeros, tmp)
        res = torch.where(torch.le(wo_z, static_zero), tmp_zeros, res)

        return res

    @classmethod
    def ggx_G1_aniso(
        cls,
        v: torch.Tensor,
        ax: torch.Tensor,
        ay: torch.Tensor,
        vz: torch.Tensor
    ) -> torch.Tensor:
        """
        If vz <= 0, return 0

        Args:
            v: shape (batch, lightnum, 3)
            ax: shape (batch, 1)
            ay: shape (batch, 1)
            vz: shape (batch, lightnum, 1)
        
        Returns:
            G1: shape (batch, lightnum, 1)
        """
        axayaz = torch.cat([ax, ay, torch.ones_like(ax)], dim=1)
        vv = v * torch.unsqueeze(axayaz, dim=1)
        G1 = 2.0 * vz / (vz + torch.norm(vv, dim=2, keepdim=True) + 1e-6)

        G1 = torch.where(
            torch.le(vz, torch.zeros_like(vz)),
            torch.zeros_like(vz),
            G1
        )
        return G1