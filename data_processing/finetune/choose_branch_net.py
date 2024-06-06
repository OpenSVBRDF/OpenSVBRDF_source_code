import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import math

class ChooseBranchNet(nn.Module):
    '''
    ChooseBranchNet defines a neural network model for selecting branches.

    functions: 
        branch_decoder: build a branch decoder model, including a series of convolutional layers, batch normalization layers and activation function layers.
                        It returns a sequential container containing all the layers and initializing the weight and bias of the convolutional layer.

        forward:        use forward propagation to obtain predicted results.
                        It returns reshaped predicted results.

    by Leyao
    '''
    def __init__(self,args):
        super(ChooseBranchNet,self).__init__()

        self.classifier_num = args["classifier_num"]

        self.layers = args["layers"]
        self.input_length = args["lighting_pattern_num"]  * (self.layers-1) 
        self.layer_width = [64,64,32,16,8,2]
        self.layer_width = [value * (self.layers-1) for value in self.layer_width]

        # construct model
        
        self.choose_branch_net_model = self.branch_decoder()
        
    def branch_decoder(self,name_prefix = "choose_branch_net_"):
        layer_stack = OrderedDict()

        layer_count = 0
        input_size = self.input_length

        for which_layer in self.layer_width[:-1]:
            output_size = which_layer
            layer_stack[name_prefix+"Conv_{}".format(layer_count)] = torch.nn.Conv1d(input_size, output_size, kernel_size=1,groups=self.layers-1)
            layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
            layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()

            layer_count+=1
            input_size = output_size

        output_size = self.layer_width[-1]
        layer_stack[name_prefix+"Conv_{}".format(layer_count)] = torch.nn.Conv1d(input_size, output_size, kernel_size=1,groups=self.layers-1)
        
        layer_stack = nn.Sequential(layer_stack)

        for m in layer_stack:
            if isinstance(m, torch.nn.Conv1d):
                m.weight.data.normal_(mean = 0, std = 0.1)
                m.bias.data.fill_(0.0)

        return layer_stack

    def forward(self,x_n):
        pred = self.choose_branch_net_model(x_n)
        pred = pred.reshape([-1,self.layers-1,2])
        pred = torch.softmax(pred,dim=-1)
        return pred
