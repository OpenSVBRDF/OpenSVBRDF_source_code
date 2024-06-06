import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import math

class LumitexelNet(nn.Module):
    '''
    LumitexelNet defines a neural network model for lumitexel.

    Functions: 
        color_latent_part: submodel for processing color information of a latent.

        latent_part:       submodel for latent part.

        lumi_net:          submodel for lumitextel.

    '''
    def __init__(self,args):
        super(LumitexelNet,self).__init__()

        self.input_length = args["lumitexel_length"]
        
        self.shape_latent_len = args["shape_latent_len"]
        self.color_latent_len = args["color_latent_len"]
        self.latent_len = self.shape_latent_len + self.color_latent_len
        # construct model
        self.latent_part_model = self.latent_part(self.input_length)
        self.color_latent_part_model = self.color_latent_part(self.input_length)
        self.lumi_net_model = self.lumi_net(self.latent_len)
    
    def color_latent_part(self,input_size,name_prefix="Color_Latent_"):
        
         
        layer_stack = OrderedDict()
        
        layer_count = 0

        output_size=input_size // 4 
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size


        output_size=1024
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size


        output_size=256
        layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=64
        layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=self.color_latent_len
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_count+=1
        input_size = output_size

        layer_stack = nn.Sequential(layer_stack)

        return layer_stack

    def latent_part(self,input_size,name_prefix="Latent_"):
        
        layer_stack = OrderedDict()
        
        layer_count = 0

        output_size=input_size // 2 
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=4096
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=2048
        layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=1024
        layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=512
        layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=256
        layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=256
        layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=self.shape_latent_len
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_count+=1
        input_size = output_size

        layer_stack = nn.Sequential(layer_stack)

        return layer_stack
    
    def lumi_net(self,input_size,name_prefix = "Lumi_"):
        
        layer_stack = OrderedDict()
        
        layer_count = 0

        output_size=128
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=256
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=256
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size


        output_size=512
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=1024
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=2048
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=2048
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=4096
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=4096
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=8192
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=self.input_length
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        
        layer_stack = nn.Sequential(layer_stack)

        return layer_stack
    

    def forward(self,net_input,input_is_latent=False,return_latent_directly=False):
        batch_size = net_input.shape[0]
        net_input = net_input.reshape([batch_size,-1])
        
        if input_is_latent is True:
            latent = net_input
        else:
            shape_latent = self.latent_part_model(net_input)
            color_latent = self.color_latent_part_model(net_input)
            latent = torch.cat([color_latent,shape_latent],dim=-1)

        if return_latent_directly:
            return latent
        

        nn_lumi = self.lumi_net_model(latent)
        
        nn_lumi  = torch.exp(nn_lumi)-1.0
        
        return latent,nn_lumi