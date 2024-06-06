import torch.nn as nn
from AUTO_lumitexel_net_inference import LumitexelNet


class PlanarScannerNet(nn.Module):
    def __init__(self, args):
        super(PlanarScannerNet, self).__init__()
        
        self.lumitexel_net = LumitexelNet(args)


    def forward(self, input, input_is_latent=True):
        nn_latent,nn_lumi = self.lumitexel_net(input, input_is_latent=input_is_latent)

        return nn_latent,nn_lumi


       

        

        


    