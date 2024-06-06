import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

class LatentNet(nn.Module):
    '''
    LatentNet defines a neural network model for extracting latent features of color and shape.

    Functions: 
        color_latent_decoder: construct a neural network model for latent features of color.

        shape_latent_decoder: construct a neural network model for latent features of shape.

    '''
    def __init__(self,args):
        super(LatentNet,self).__init__()
        self.leaf_nodes_num = args["leaf_nodes_num"]
        self.cam_num = args["cam_num"]
        self.input_length = args["lighting_pattern_num"] * self.cam_num * self.leaf_nodes_num
        
        
        self.color_latent_len = args["color_latent_len"]
        self.shape_latent_len = args["shape_latent_len"] 
        self.latent_len = self.shape_latent_len + self.color_latent_len

        # construct model
        self.shape_layer_width = [256,1024,3072,1024,256,self.shape_latent_len]
        self.shape_layer_width = [value * self.leaf_nodes_num for value in self.shape_layer_width]

        self.color_layer_width = [256,1024,512,64,self.color_latent_len]
        self.color_layer_width = [value * self.leaf_nodes_num for value in self.color_layer_width]
        
        self.shape_latent_net_model = self.shape_latent_decoder(self.input_length)
        self.color_latent_net_model = self.color_latent_decoder(self.input_length)
        
    def color_latent_decoder(self,input_size,name_prefix = "color_latent_net_"):
        layer_stack = OrderedDict()

        layer_count = 0
        input_size = self.input_length

        for which_layer in self.color_layer_width[:-1]:
            output_size = which_layer
            layer_stack[name_prefix+"Conv_{}".format(layer_count)] = torch.nn.Conv1d(input_size, output_size, kernel_size=1,groups=self.leaf_nodes_num)
            if which_layer > 1:
                layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
            layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
            layer_count+=1
            input_size = output_size
        
        output_size = self.color_layer_width[-1]
        layer_stack["Linear_{}".format(layer_count)] = torch.nn.Conv1d(input_size, output_size, kernel_size=1, groups=self.leaf_nodes_num)
        
        layer_stack = nn.Sequential(layer_stack)

        return layer_stack

    def shape_latent_decoder(self,input_size,name_prefix = "shape_latent_net_"):
        layer_stack = OrderedDict()

        layer_count = 0
        input_size = self.input_length

        for which_layer in self.shape_layer_width[:-1]:
            output_size = which_layer
            layer_stack[name_prefix+"Conv_{}".format(layer_count)] = torch.nn.Conv1d(input_size, output_size, kernel_size=1,groups=self.leaf_nodes_num)
            if which_layer > 2:
                layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
            layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
            layer_count+=1
            input_size = output_size
        
        output_size = self.shape_layer_width[-1]
        layer_stack["Linear_{}".format(layer_count)] = torch.nn.Conv1d(input_size, output_size, kernel_size=1, groups=self.leaf_nodes_num)
        
        layer_stack = nn.Sequential(layer_stack)

        return layer_stack

    
    def forward(self,measurements):
        batch_size = measurements.size()[0]
        x_n = measurements.unsqueeze(dim=1).unsqueeze(dim=-1).repeat(1,self.leaf_nodes_num,1,1).reshape([batch_size,-1,1])
        
    
        shape_latent = self.shape_latent_net_model(x_n)
        color_latent = self.color_latent_net_model(x_n)
        
        
        latent = torch.cat([color_latent,shape_latent],dim=1)
        latent = latent.reshape([batch_size,self.leaf_nodes_num,self.latent_len])
         
        
        return latent