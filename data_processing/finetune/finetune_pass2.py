'''
 This is the experimental code for paper ``Ma et al., OpenSVBRDF: A Database of Measured Spatially-Varying Reflectance, SIGGRAPH Asia 2023``.
 This script is suboptimal and experimental.
 There may be redundant lines and functionalities.

 Xiaohe Ma, 2024/02
'''

'''
 This script implements the second step of the per-pixel fine-tuning process.

'''



import numpy as np
import argparse
import torch
import sys
import os
from datetime import datetime
from torch.autograd import Variable 
import AUTO_planar_scanner_net_inference



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_for_server_root",default="data_for_server/")
    parser.add_argument("--lighting_pattern_num",type=int,default="128")
    parser.add_argument("--finetune_use_num",type=int,default="128")
    parser.add_argument("--batchsize",type=int,default=100)
    parser.add_argument('--thread_ids', nargs='+', type=int,default=[5])
    parser.add_argument('--gpu_id', type=int,default=0)
    parser.add_argument("--need_dump",action="store_true")
    parser.add_argument("--total_thread_num",type=int,default=24)
    parser.add_argument("--tex_resolution",type=int,default=512)
    parser.add_argument("--cam_num",type=int,default=2)
    parser.add_argument("--main_cam_id",type=int,default=0)
    parser.add_argument("--model_file",type=str,default="../../model/model_state_450000.pkl")
    parser.add_argument("--pattern_file",type=str,default="../../model/opt_W.bin")
    parser.add_argument("--shape_latent_len",type=int,default=2)
    parser.add_argument("--color_latent_len",type=int,default=2)
    parser.add_argument("--save_lumi",action="store_true")

    args = parser.parse_args()
    compute_device = torch.device("cuda:{}".format(args.gpu_id))
    
    train_configs = {}
    train_configs["training_device"] = args.gpu_id
    train_configs["lumitexel_length"] = 64*64*3
    train_configs["shape_latent_len"] = args.shape_latent_len
    train_configs["color_latent_len"] = args.color_latent_len
    
    all_latent_len = args.shape_latent_len + args.color_latent_len * 3

    pretrained_dict = torch.load(args.model_file, map_location=compute_device)
    inference_net = AUTO_planar_scanner_net_inference.PlanarScannerNet(train_configs)

    m_len_perview = 3
    
    something_not_found = False
    model_dict = inference_net.state_dict()
    for k,_ in model_dict.items():
        if k not in pretrained_dict:
            print("not found:", k)
            something_not_found = True
    if something_not_found:
        exit()

    model_dict = inference_net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    inference_net.load_state_dict(model_dict)
    for p in inference_net.parameters():
        p.requires_grad=False
    inference_net.to(compute_device)
    
    inference_net.eval()


    for which_test_thread in args.thread_ids:
        data_root = args.data_for_server_root+"{}/".format(which_test_thread)
        print(data_root)

        lighting_patterns_np = np.fromfile(args.pattern_file,np.float32).reshape([args.lighting_pattern_num,-1,3])[:args.finetune_use_num,:,0].reshape([1,args.finetune_use_num,-1])
        
        lighting_patterns = torch.from_numpy(lighting_patterns_np).to(compute_device) 
        
        log_path = data_root+"lumi_imgs/"
        os.makedirs(log_path,exist_ok=True)

        
        measurements = torch.from_numpy(np.fromfile(data_root+"gt_measurements_{}.bin".format(args.tex_resolution),np.float32)).to(compute_device).reshape((-1,args.lighting_pattern_num,m_len_perview))[:,:args.finetune_use_num,:]
            
        pf_pass1_latent = np.fromfile(data_root+"pass1_latent_{}.bin".format(args.tex_resolution),np.float32).reshape([-1,all_latent_len])
        latent_num = pf_pass1_latent.shape[0]
        pf_pass1_latent = torch.from_numpy(pf_pass1_latent).to(compute_device)

        assert measurements.shape[0] == latent_num,"some data are corrupted"
        sample_num = measurements.shape[0]
        
        pf_result = open(data_root+"pass2_latent_{}.bin".format(args.tex_resolution),"wb")

        ptr = 0
        
        color_optimize_step = 50
        color_lr = 0.05


        while True:
            
            if ptr % 30000 == 0:
                start = datetime.now()
                print(f"PASS 2 [{which_test_thread}]/{ptr}/{sample_num}   {start}")

            tmp_measurements = measurements[ptr:ptr+args.batchsize]
            
            cur_batchsize = tmp_measurements.shape[0]
            if cur_batchsize == 0: 
                print("break because all done.")
                break


            tmp_pass1_color_latent = pf_pass1_latent[ptr:ptr+cur_batchsize,:args.color_latent_len*3]
            tmp_pass1_shape_latent = pf_pass1_latent[ptr:ptr+cur_batchsize,args.color_latent_len*3:]

            latent_start = datetime.now()
            batch_collector = []
            pass2_color_lumi = []

            for which_channel in range(3):
                tmp_channel_measurements = tmp_measurements[:,:,which_channel].reshape([cur_batchsize,-1])
                
                tmp_color_guess = tmp_pass1_color_latent[:,args.color_latent_len*which_channel:args.color_latent_len*(which_channel+1)].clone()

                tmp_color_guess = Variable(tmp_color_guess,requires_grad=True)

                color_optimizer = torch.optim.Adam([tmp_color_guess,], lr = color_lr)

                loss_step = []
                loss_precent = []
                for step in range(color_optimize_step):
                    tmp_color_shape_guess = torch.cat([tmp_color_guess,tmp_pass1_shape_latent],dim=-1)
                    _,tmp_channel_lumi = inference_net(tmp_color_shape_guess,input_is_latent=True)
                    tmp_channel_lumi = torch.max(torch.zeros_like(tmp_channel_lumi),tmp_channel_lumi)

                    tmp_lumi_measurements = torch.sum(lighting_patterns*tmp_channel_lumi.unsqueeze(dim=1),dim=-1).reshape([cur_batchsize,-1])
                    
                    color_loss = torch.nn.functional.mse_loss(torch.pow(tmp_lumi_measurements,1/2.0), torch.pow(tmp_channel_measurements,1/2.0),reduction='sum')

                    color_optimizer.zero_grad()
                    color_loss.backward() 
                    color_optimizer.step()

                    loss_step.append(color_loss)
                    if step >= 5:
                        tmp_loss_percent = color_loss/loss_step[step-5]*100
                        loss_precent.append(tmp_loss_percent)
                        if len(loss_precent) >= 5:
                            if torch.mean(torch.stack(loss_precent[-5:],dim=0)) > 95.0:
                                break
                        
                tmp_color_guess = tmp_color_guess.detach()
                
                batch_collector.append(tmp_color_guess)
                pass2_color_lumi.append(tmp_channel_lumi)

            batch_collector.append(tmp_pass1_shape_latent)
            batch_collector = torch.cat(batch_collector,dim=-1)
            pass2_color_lumi = torch.stack(pass2_color_lumi,dim=-1)

            batch_collector.cpu().numpy().astype(np.float32).tofile(pf_result)  

            ptr = ptr+cur_batchsize

        pf_result.close()

    print("done.")
        
        















        
        

         
        
        