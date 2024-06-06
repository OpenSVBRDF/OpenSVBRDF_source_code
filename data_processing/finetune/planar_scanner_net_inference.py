import torch.nn as nn
import torch.utils.data
import numpy as np
import math

from choose_branch_net import ChooseBranchNet

from latent_net import LatentNet

class PlanarScannerNet(nn.Module):
    '''
    PlanarScannerNet implements a neural network model for finetuning.

    Functions: 
        precompute_table: define a table for searching on initialization.

        forward:          forward propagation method for input data.

    '''
    def __init__(self, args):
        super(PlanarScannerNet, self).__init__()
        ########################################
        ## parse configuration               ###
        ########################################
        self.args = args
        self.training_device = args["training_device"]
        self.layers = args["layers"]
        
        self.m_len = args["m_len"]
        self.cam_num = args["cam_num"]
        self.main_cam_id = args["main_cam_id"]
        self.batch_size = args["batch_size"]
        self.sublings_num = 2
        self.lumitexel_length = args["lumitexel_length"]
        ########################################
        ## loading setup configuration       ###
        ########################################
        self.leaf_nodes_num = int(math.pow(2,self.layers-1))
        args["classifier_num"] = self.leaf_nodes_num - 1
        args["leaf_nodes_num"] = self.leaf_nodes_num
        
        self.ptr = self.leaf_nodes_num - 1
        self.return_all_leaf = False
        ########################################
        ## define net modules                ###
        ########################################
        self.l2_loss_fn = torch.nn.MSELoss(reduction='sum')

        self.latent_net = LatentNet(args)
        self.choose_branch_net = ChooseBranchNet(args)

        self.search_table = self.precompute_table()
        
        self.leaf_indices = torch.arange(self.leaf_nodes_num).long()
        self.layer_indices = self.search_table[:,:,0].long()
        self.base = torch.from_numpy(np.array([16,8,4,2,1])).long().to(self.training_device)
                
    def precompute_table(self):
        search_table = []
        for which_node in range(self.ptr, self.ptr+self.leaf_nodes_num):
            tmp_router = []
            subling = which_node
            parent = (which_node - 1) // self.sublings_num
            while parent >= 0:
                tmp_router.append(parent)
                tmp_router.append((subling-1)%self.sublings_num)
                subling = parent
                parent = (parent-1) // self.sublings_num
                
            search_table.append(torch.as_tensor(tmp_router))
        
        search_table = torch.stack(search_table,dim=0).reshape([self.leaf_nodes_num,self.layers-1,2]) 
        
        return search_table
    
    def calculate_loss(self,pred,label):
        tmp_loss = pred - label
        tmp_loss = torch.sum(tmp_loss*tmp_loss,dim=1,keepdims=True).reshape(self.batch_size,1)
        return tmp_loss.clone()

            
    def forward(self, batch_data,call_type="train"):
        measurements = batch_data
        batch_size = measurements.shape[0]
        self.batch_indices = torch.arange(batch_size)[:,None]
        
        measurements_main_cam = measurements[:,self.main_cam_id] 
        
        expand_measurements = measurements_main_cam.unsqueeze(dim=1).repeat(1,self.layers-1,1).reshape([batch_size,-1]).unsqueeze(dim=-1)
        measurements = torch.cat([measurements[:,0],measurements[:,1]],dim=-1)

        clean_logits = self.choose_branch_net(expand_measurements)
        
        expand_weight = []
        for which_layer in range(self.layers-1):
            tmp_choose_net_output = clean_logits[:,which_layer].unsqueeze(dim=1).repeat(1,int(math.pow(self.sublings_num,which_layer)),1)
            
            expand_weight.append(tmp_choose_net_output)

        all_classifier_weight = torch.cat(expand_weight,dim=1)
        
        
        all_leaf_index = self.search_table.clone()
        all_leaf_index = all_leaf_index.repeat(batch_size,1,1,1)
        
        all_classifier_weight = all_classifier_weight.reshape([batch_size,-1,2]).unsqueeze(dim=1).repeat(1,self.leaf_nodes_num,1,1) 
        
        all_leaf_weight = all_classifier_weight.reshape([batch_size*self.leaf_nodes_num,-1,2])
        
        all_leaf_index = all_leaf_index.reshape([batch_size*self.leaf_nodes_num,-1,2])

        self.batch_indices = torch.arange(batch_size*self.leaf_nodes_num)[:,None]
        
        all_leaf_weight = all_leaf_weight[self.batch_indices,all_leaf_index[:,:,0]]
       
        
        all_leaf_weight = all_leaf_weight.reshape([-1,2])
        
        all_leaf_index = all_leaf_index[:,:,1].reshape([-1,1])
        self.batch_indices = torch.arange(all_leaf_index.shape[0])[:,None]

        all_leaf_weight = all_leaf_weight[self.batch_indices,all_leaf_index].reshape([batch_size,self.leaf_nodes_num,self.layers-1])
        
        all_leaf_weight = torch.prod(all_leaf_weight,dim=2)
        
        
        gates = all_leaf_weight
        
        ################### BALANCE LOSS ####################

        router = torch.argsort(gates,dim=-1)[:,-1]

        router = router.unsqueeze(dim=-1)
        
        nn_latent = self.latent_net(measurements)
        nn_latent = nn_latent.reshape([batch_size,self.leaf_nodes_num,-1])
        
        select_nn_latent = torch.zeros_like(nn_latent[:,0])
        
        for which_node in range(self.leaf_nodes_num):
            
            tmp_router_idx = torch.where(router == which_node)[0]
            if tmp_router_idx.shape[0] > 0:
                select_nn_latent[tmp_router_idx] = nn_latent[tmp_router_idx][:,which_node]

        
        term_map = {
            "nn_latent":select_nn_latent.detach().cpu().numpy(),
        }

        return term_map
