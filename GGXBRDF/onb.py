import torch
import torch.nn.functional as F


class ONB(object):

    def __init__(self, batch_size : int):
        """
        uvw - xyz - tbn 
        """
        self.batch_size = batch_size
        self.axis = torch.zeros(batch_size, 3, 3)    # (batch, 3, 3)
        self.axis[:, 0, 0] = 1
        self.axis[:, 1, 1] = 1
        self.axis[:, 2, 2] = 1
    
    def u(self) -> torch.Tensor:
        """
        Returns:
            u axis of shape (batch, 3)
        """
        return self.axis[:, 0, :]
    
    def v(self) -> torch.Tensor:
        return self.axis[:, 1, :]
    
    def w(self) -> torch.Tensor:
        return self.axis[:, 2, :]

    def inverse_transform(self, p : torch.Tensor) -> torch.Tensor:
        """
        Convert local coordinate(in onb) back to global 
        coordinate(onb in).

        Args:
            p: local coordinates of shape (batch, 3) or (batch, N, 3)
        
        Returns:
            global coordinates of shape (batch, 3) or (batch, N, 3)
        """
        assert(self.batch_size == p.size(0))
        assert(len(p.size()) in [2, 3])
        if len(p.size()) == 2:
            return p[:, 0:1] * self.u() + p[:, 1:2] * self.v() + p[:, 2:3] * self.w()
        elif len(p.size()) == 3:
            u = self.u().unsqueeze(1)
            v = self.v().unsqueeze(1)
            w = self.w().unsqueeze(1)
            return p[:, :, [0]] * u + p[:, :, [1]] * v + p[:, :, [2]] * w

    def transform(self, p : torch.Tensor) -> torch.Tensor:
        """
        Convert global coordinate(onb in) to local
        coordinate(in onb).

        Args:
            p: global coordinates of shape (batch, 3) or (batch, lightnum, 3)

        Returns:
            local coordinates of shape (batch, 3) or (batch, lightnum, 3)
        """
        assert(self.batch_size == p.size(0))
        assert(len(p.size()) in [2, 3])
        if len(p.size()) == 2:
            x = torch.sum(p * self.u(), dim=1, keepdim=True)
            y = torch.sum(p * self.v(), dim=1, keepdim=True)
            z = torch.sum(p * self.w(), dim=1, keepdim=True)
            return torch.cat([x, y, z], dim=1) 
        elif len(p.size()) == 3:
            lightnum = p.size(1)
            u = self.u().unsqueeze(1).repeat(1, lightnum, 1)
            v = self.v().unsqueeze(1).repeat(1, lightnum, 1)
            w = self.w().unsqueeze(1).repeat(1, lightnum, 1)
            x = torch.sum(p * u, dim=2, keepdim=True)
            y = torch.sum(p * v, dim=2, keepdim=True)
            z = torch.sum(p * w, dim=2, keepdim=True)
            return torch.cat([x, y, z], dim=2)

    def build_from_ntb(
        self,
        n : torch.Tensor, 
        t : torch.Tensor,
        b : torch.Tensor,
    ) -> None:
        """
        Args:
            n, t, b: The local frame, of shape (batch, 3)
        """
        batch_size = n.size(0)
        self.axis = torch.zeros((batch_size, 3, 3)).to(n.device)
        self.axis[:, 2, :] = F.normalize(n, dim=1)
        self.axis[:, 1, :] = F.normalize(b, dim=1)
        self.axis[:, 0, :] = F.normalize(t, dim=1)

    def build_from_w(self, normal : torch.Tensor) -> None:
        """
        Build the local frame based on the normal.

        Args:
            normal: The normal coordinates of shape (batch, 3)
        """
        assert(self.batch_size == normal.size(0))
        device = normal.device
        n = F.normalize(normal, dim=1)
        nz = n[:, [2]]
        batch_size = n.shape[0]

        constant_001 = torch.zeros_like(normal).to(device)
        constant_001[:, 2] = 1.0
        constant_100 = torch.zeros_like(normal).to(device)
        constant_100[:, 0] = 1.0

        nz_notequal_1 = torch.gt(torch.abs(nz - 1.0), 1e-6)
        nz_notequal_m1 = torch.gt(torch.abs(nz + 1.0), 1e-6)

        
        t = torch.where(nz_notequal_1 & nz_notequal_m1, constant_001, constant_100)
        # Optix version
        # b = F.normalize(torch.cross(normal, t), dim=1)
        # t = torch.cross(b, normal)
        # Original pytorch version
        t = F.normalize(torch.cross(t, normal), dim=1)
        b = torch.cross(n, t)

        self.axis = torch.zeros((batch_size, 3, 3)).to(device)
        self.axis[:, 2, :] = n
        self.axis[:, 1, :] = b
        self.axis[:, 0, :] = t
    
    def rotate_frame(self, theta : torch.Tensor) -> None:
        """
        Rotate local frame along the normal axis

        Args:
            theta: the degrees of counterclockwise rotation, of shape (batch, 1)
        """
        assert(self.batch_size == theta.size(0))
        n = self.w()
        t = self.u()
        b = self.v()

        t = F.normalize(t * torch.cos(theta) + b * torch.sin(theta), dim=1)
        b = F.normalize(torch.cross(n, t), dim=1)
        self.axis = torch.zeros((self.batch_size, 3, 3)).to(theta.device)
        self.axis[:, 0, :] = t
        self.axis[:, 1, :] = b
        self.axis[:, 2, :] = n

    def _back_hemi_octa_map(self, n_2d : torch.Tensor) -> torch.Tensor:
        """
        The original normal is (0, 0, 1), we should use this method to
        perturb the original normal to get a new normal and then build
        a new local frame based on the new normal.

        Args:
            n_2d: shape (batch, 2)
        
        Returns:
            local_n: shape (batch, 3), which is define in geometry local
                frame.
        """
        p = (n_2d - 0.5) * 2.0
        resultx = (p[:, [0]] + p[:, [1]]) * 0.5
        resulty = (p[:, [1]] - p[:, [0]]) * 0.5
        resultz = 1.0 - torch.abs(resultx) - torch.abs(resulty)
        result = torch.cat([resultx, resulty, resultz], dim=1)
        return F.normalize(result, dim=1)

    def hemi_octa_map(self, dir : torch.Tensor) -> torch.Tensor:
        """
        Args:
            dir: shape (batch, 3)
        
        Returns:
            n2d: shape (batch, 2), which is define in circle coordinate
        """
        high_dim = False
        batch_size = dir.shape[0]

        if len(dir.shape) > 2:
            high_dim = True
            dir = dir.reshape(-1, 3)

        p = dir/torch.sum(torch.abs(dir), dim=1, keepdim=True) # (batch,3)
        n_2d = torch.cat([p[:,[0]] - p[:,[1]],p[:,[0]] + p[:, [1]]],dim=1) * 0.5 + 0.5
        if high_dim:
            n_2d = n_2d.reshape(batch_size, -1, 2)
        return n_2d

    def build_from_n2d(self, n_2d : torch.Tensor, theta : torch.Tensor) -> None:
        """
        Args:
            n_2d: tensor of shape (batch, 2). the param defines how 
                to perturb local normal.
            theta: the degrees of rotation of tangent.
        """
        assert(self.batch_size == n_2d.size(0))

        local_n = self._back_hemi_octa_map(n_2d)
        self.build_from_w(local_n)
        self.rotate_frame(theta)
