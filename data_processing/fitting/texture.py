import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class Texture(nn.Module):
    '''
    Texture implements simple texture mapping.

    Methods: 
        forward:          Defines forward propagation. Performs texture mapping of the input UV coordinates.
        reset_parameters: Resets the parameters using Xavier initialization.
        set_parameters:   Sets the texture map parameters.
    '''
    
    def __init__(self, width, height, feature_num):
        super(Texture, self).__init__()
        self.width = width
        self.height = height
        self.feature_num = feature_num
        self.params = nn.Parameter(torch.Tensor(1, feature_num, width, height))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.params.data)

    def set_parameters(self, params, requires_grad):
        param_data = params.permute(2, 0, 1).unsqueeze(0).to(self.params.device)
        self.params.data.copy_(param_data)
        self.params.requires_grad = requires_grad

    def forward(self, uv_):
        batch = uv_.size(0)
        uv = uv_ * 2.0 - 1.0 
        
        uv = uv.view(1, -1, 1, 2)
        y = F.grid_sample(self.params, uv, align_corners=True)

        y = y.view(-1, batch).transpose(1, 0)
        return y