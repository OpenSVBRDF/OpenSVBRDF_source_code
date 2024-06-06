'''
 This is the experimental code for paper ``Ma et al., OpenSVBRDF: A Database of Measured Spatially-Varying Reflectance, SIGGRAPH Asia 2023``.
 This script is suboptimal and experimental.
 There may be redundant lines and functionalities.

 Xiaohe Ma, 2024/02
'''

'''
This script implements the first step of the per-pixel fine-tuning process.

'''

import numpy as np
import argparse
import torch
import os
from datetime import datetime
from torch.autograd import Variable 
import AUTO_planar_scanner_net_inference


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_for_server_root",default="data_for_server/")
    parser.add_argument("--lighting_pattern_num",type=int,default=64)
    parser.add_argument("--finetune_use_num",type=int,default=64)
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

    torch.set_printoptions(precision=3)

    compute_device = torch.device(f"cuda:{args.gpu_id}")
    
    train_configs = {}
    train_configs["training_device"] = args.gpu_id
    train_configs["lumitexel_length"] = 64*64*3
    train_configs["shape_latent_len"] = args.shape_latent_len
    train_configs["color_latent_len"] = args.color_latent_len
    
    latent_len = args.shape_latent_len + args.color_latent_len

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
        data_root = args.data_for_server_root+f"{which_test_thread}/"
        print(data_root)

        lighting_patterns_np = np.fromfile(args.pattern_file,np.float32).reshape([args.lighting_pattern_num,-1,3])[:args.finetune_use_num,:,0].reshape([1,args.finetune_use_num,-1])
        
        lighting_patterns = torch.from_numpy(lighting_patterns_np).to(compute_device)
        
        log_path = data_root+"lumi_imgs/"
        os.makedirs(log_path,exist_ok=True)

        
        measurements = torch.from_numpy(np.fromfile(data_root+f"gt_measurements_{args.tex_resolution}.bin",np.float32)).to(compute_device).reshape((-1,args.lighting_pattern_num,m_len_perview))[:,:args.finetune_use_num,:]
        
        
        pf_nn_latent = np.fromfile(data_root+f"latent_{args.tex_resolution}.bin",np.float32).reshape([-1,latent_len])
        latent_num = pf_nn_latent.shape[0]

        assert measurements.shape[0] == latent_num,"some data are corrupted"
        sample_num = measurements.shape[0]
        
        pf_result_grey = open(data_root+f"pass1_latent_{args.tex_resolution}.bin","wb")
        pf_result_nn = open(data_root+f"pass0_latent_{args.tex_resolution}.bin","wb")
        
        ptr = 0
        optimize_step = 500
        lr = 0.02

        print("finetune latent...")

        while True:
            if ptr % 30000 == 0:
                start = datetime.now()
                print(f"PASS 1 [{which_test_thread}]/{ptr}/{sample_num}   {start}")

            tmp_measurements = measurements[ptr:ptr+args.batchsize]
            tmp_measurements_mean = tmp_measurements.mean(dim=-1)

            cur_batchsize = tmp_measurements.shape[0]
            if cur_batchsize == 0: 
                print("break because all done.")
                break
            
            tmp_nn_latent = pf_nn_latent[ptr:ptr+cur_batchsize]
            tmp_nn_color_latent = pf_nn_latent[ptr:ptr+cur_batchsize,:args.color_latent_len]
            tmp_nn_shape_latent = pf_nn_latent[ptr:ptr+cur_batchsize,args.color_latent_len:]
            tmp_nn_latent_3c = np.concatenate([tmp_nn_color_latent, tmp_nn_color_latent,tmp_nn_color_latent,tmp_nn_shape_latent],axis=1)
            tmp_nn_latent_3c.astype(np.float32).tofile(pf_result_nn)

            tmp_x_guess = torch.cuda.FloatTensor(tmp_nn_latent,device=compute_device)
            tmp_x_guess = Variable(tmp_x_guess,requires_grad=True)
            optimizer = torch.optim.Adam([tmp_x_guess,], lr = lr)

            latent_start = datetime.now()

            loss_step = []
            loss_precent = []
            for step in range(optimize_step):
                _,tmp_lumi = inference_net(tmp_x_guess,input_is_latent=True)
                tmp_lumi = torch.max(torch.zeros_like(tmp_lumi),tmp_lumi)

                tmp_lumi_measurements = torch.sum(lighting_patterns*tmp_lumi.unsqueeze(dim=1),dim=-1).reshape([cur_batchsize,-1])
                
                loss = torch.nn.functional.mse_loss(torch.pow(tmp_lumi_measurements,1/2.0), torch.pow(tmp_measurements_mean,1/2.0),reduction='sum')
                        
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()

                loss_step.append(loss)
                if step >= 10:
                    tmp_loss_percent = loss/loss_step[step-10]*100
                    loss_precent.append(tmp_loss_percent)
                    if len(loss_precent) >= 5:
                        if torch.mean(torch.stack(loss_precent[-5:],dim=0)) > 95.0:
                            break
                
            tmp_x_guess = tmp_x_guess.detach()

            batch_collector = []

            nn_color_lumi = []

            tmp_shape_latent = tmp_x_guess[:,args.color_latent_len:]
            
            tmp_color_latent = tmp_x_guess[:,:args.color_latent_len].unsqueeze(dim=1).repeat(1,3,1).reshape([-1,args.color_latent_len*3])
            grey_latent_3channel = torch.cat([tmp_color_latent,tmp_shape_latent],dim=-1)
            grey_latent_3channel.cpu().numpy().astype(np.float32).tofile(pf_result_grey)
                    

            ptr = ptr+cur_batchsize

        pf_result_grey.close()
        pf_result_nn.close()

    print("done.")
