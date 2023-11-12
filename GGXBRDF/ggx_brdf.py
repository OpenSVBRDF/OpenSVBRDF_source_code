import math
import torch
import torch.nn.functional as F
from onb import ONB


class GGX_BRDF(object):

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
        wi_local = local_onb.transform(wi)    # (batch, lightnum, 3)
        wo_local = local_onb.transform(wo)    # (batch, 3)

        meta = {}

        a = torch.unsqueeze(pd / math.pi, dim=1)  # (batch, 1, 1)
        b = cls.ggx_brdf_aniso(wi_local, wo_local, ax, ay, specular_component)  # (batch, lightnum, 1)
        ps = torch.unsqueeze(ps, dim=1)  # (batch, 1, channel)

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

        wi_z = wi[:, :, [2]]  # (batch, lightnum, 1)
        wo_z = wo[:, :, [2]]
        denom = 4 * wi_z * wo_z  # (batch, lightnum, 1)
        vhalf = F.normalize(wi + wo, dim=2)  # (batch, lightnum, 3)

        # F
        tmp = torch.clamp(1.0 - torch.sum(wi * vhalf, dim=2, keepdim=True), 0, 1)
        F0 = 0.04
        Fresnel = F0 + (1 - F0) * tmp * tmp * tmp * tmp * tmp

        # D
        axayaz = torch.unsqueeze(torch.cat([ax, ay, torch.ones_like(ax)], dim=1),
                                dim=1)  # (batch, 1, 3)
        vhalf = vhalf / (axayaz + 1e-6)  # (batch, lightnum, 3)
        vhalf_norm = torch.norm(vhalf, dim=2, keepdim=True)
        length = vhalf_norm * vhalf_norm  # (batch, lightnum, 1)
        D = 1.0 / (math.pi * torch.unsqueeze(ax, dim=1) *
                torch.unsqueeze(ay, dim=1) * length * length)

        # G
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
        axayaz = torch.cat([ax, ay, torch.ones_like(ax)], dim=1)    # (batch, 3)
        vv = v * torch.unsqueeze(axayaz, dim=1)                     # (batch, lightnum, 3)
        G1 = 2.0 * vz / (vz + torch.norm(vv, dim=2, keepdim=True) + 1e-6)

        # If vz < 0, G1 will be zero.
        G1 = torch.where(
            torch.le(vz, torch.zeros_like(vz)),
            torch.zeros_like(vz),
            G1
        )
        return G1