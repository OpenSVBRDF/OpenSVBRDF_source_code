from typing import List
import torch.nn as nn
from collections import OrderedDict
import torch.nn.init as init
import math


class MLPModel(nn.Module):
    '''
    MLPModel defines a simple multilayer perceptron model.

    Methods:
        __init__:          Initializes the model with the input dimension list. Linear layers are added. Normalization layers and activation functions are included if specified.
        forward:           Defines forward propagation.
        reset_parameters:  Resets the parameters of the linear layers using Kaiming initialization.
    '''
    
    def __init__(
        self,
        dim_list: List[int],
        normalizaiton='BN',
        activation='leaky_relu',
        output_activation=None,
    ):
        super(MLPModel, self).__init__()

        layers = OrderedDict()
        for i in range(len(dim_list)-2):
            layers[f"Linear_{i}"] = nn.Linear(dim_list[i], dim_list[i+1])
            if normalizaiton == "BN":
                layers["Batchnorm_{}".format(i)] = nn.BatchNorm1d(dim_list[i+1])
            elif normalizaiton == "LN":
                layers["Layernorm_{}".format(i)] = nn.LayerNorm(dim_list[i+1])
            if activation == 'leaky_relu':
                layers[f"LeakyRelu_{i}"] = nn.LeakyReLU(0.2, inplace=True)
            elif activation == 'relu':
                layers[f"Relu_{i}"] = nn.ReLU()

        i = len(dim_list)-2
        layers[f"Linear_{i}"] = nn.Linear(dim_list[i], dim_list[i+1])

        if output_activation is not None and output_activation == "sigmoid":
            layers["Sigmoid"] = nn.Sigmoid()

        self.decode_part = nn.Sequential(layers)

    def forward(self, x):
        out = self.decode_part(x)
        return out

    def reset_parameters(self) -> None:
        
        def reset(m):
            if type(m) == nn.Linear:
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    init.uniform_(m.bias, -bound, bound)

        self.decode_part.apply(reset)
