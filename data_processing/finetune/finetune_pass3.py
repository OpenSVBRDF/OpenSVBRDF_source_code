'''
 This is the experimental code for paper ``Ma et al., OpenSVBRDF: A Database of Measured Spatially-Varying Reflectance, SIGGRAPH Asia 2023``.
 This script is suboptimal and experimental.
 There may be redundant lines and functionalities.

 Xiaohe Ma, 2024/02
'''

'''
 This script implements the third step of the per-pixel fine-tuning process.

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
    parser.add_argument("--pattern_file",type=str,default="../../model/opt_W.bin.pkl")
    parser.add_argument("--shape_latent_len",type=int,default=2)
    parser.add_argument("--color_latent_len",type=int,default=2)
    parser.add_argument("--save_lumi",action="store_true")
    parser.add_argument("--if_continue",type=int,default=0)

    args = parser.parse_args()
    compute_device = torch.device("cuda:{}".format(args.gpu_id))
    
    
    train_configs = {}
    train_configs["training_device"] = args.gpu_id
    train_configs["lumitexel_length"] = 64*64*3
    train_configs["shape_latent_len"] = args.shape_latent_len
    train_configs["color_latent_len"] = args.color_latent_len
    
    latent_len = args.shape_latent_len + args.color_latent_len
    
    all_latent_len = 3*args.color_latent_len + args.shape_latent_len

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

        lighting_patterns_np = np.fromfile(args.pattern_file,np.float32).reshape([args.lighting_pattern_num,-1,3])[:args.finetune_use_num,:,:].reshape([1,args.finetune_use_num,-1,3])
        
        lighting_patterns = torch.from_numpy(lighting_patterns_np).to(compute_device)
        
        log_path = data_root+"lumi_imgs/"
        os.makedirs(log_path,exist_ok=True)

        measurements = torch.from_numpy(np.fromfile(data_root+f"gt_measurements_{args.tex_resolution}.bin",np.float32)).to(compute_device).reshape((-1,args.lighting_pattern_num,m_len_perview))[:,:args.finetune_use_num,:]

        if os.path.isfile(data_root+f"pass3_latent_{args.tex_resolution}.bin") and args.if_continue:
            finetune_latent = np.fromfile(data_root+f"pass3_latent_{args.tex_resolution}.bin",np.float32).reshape([-1,all_latent_len])
            print("load from 3")
        else:
            finetune_latent = np.fromfile(data_root+f"pass2_latent_{args.tex_resolution}.bin",np.float32).reshape([-1,all_latent_len])
            print("load from 2")
        
        assert measurements.shape[0] == finetune_latent.shape[0],"some data are corrupted"
        sample_num = finetune_latent.shape[0]
        texel_sequence = np.arange(sample_num)
        
        pf_result = open(data_root+f"pass3_latent_{args.tex_resolution}.bin","wb")
        pf_result_lumi = open(data_root+"finetune_lumi.bin","wb")

        optimize_step = 300 if args.if_continue else 500
        
        lr = 0.02
        ptr = 0
        while True:
            if ptr % 30000 == 0:
                start = datetime.now()
                print(f"PASS 3 [{which_test_thread}]/{ptr}/{sample_num}   {start}")
                
            tmp_sequence = texel_sequence[ptr:ptr+args.batchsize]
            tmp_seq_size = tmp_sequence.shape[0]
            
            if tmp_seq_size == 0:
                break
            
            tmp_measurements = measurements[tmp_sequence]
            
            tmp_x_guess = torch.cuda.FloatTensor(finetune_latent[tmp_sequence],device=compute_device)
            tmp_x_guess = Variable(tmp_x_guess,requires_grad=True)
            optimizer = torch.optim.Adam([tmp_x_guess,], lr = lr)

            loss_step = []
            loss_precent = []
            for opt_step in range(optimize_step):
                
                color_latent = tmp_x_guess[:,:3 * args.color_latent_len].reshape([tmp_seq_size,3,args.color_latent_len])
                shape_latent = tmp_x_guess[:,3 * args.color_latent_len:].reshape([tmp_seq_size,1,args.shape_latent_len]).repeat(1,3,1)
                
                color_shape_latent = torch.cat([color_latent,shape_latent],dim=-1)
                color_shape_latent = color_shape_latent.reshape([tmp_seq_size*3,latent_len])
                
                _,tmp_nn_lumi = inference_net(color_shape_latent,input_is_latent=True)
                tmp_nn_lumi = torch.max(torch.zeros_like(tmp_nn_lumi),tmp_nn_lumi)
                tmp_nn_lumi = tmp_nn_lumi.reshape([tmp_seq_size,3,-1]).permute(0,2,1).unsqueeze(dim=1)
                
                tmp_lumi_measurements = torch.sum(lighting_patterns*tmp_nn_lumi,dim=2).reshape([tmp_seq_size,args.finetune_use_num,3])
                
                loss = torch.nn.functional.mse_loss(torch.pow(tmp_measurements,1/2), torch.pow(tmp_lumi_measurements,1/2),reduction='sum')
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_step.append(loss)
                if opt_step >= 10 and not args.if_continue:
                    tmp_loss_percent = loss/loss_step[opt_step-10]*100
                    loss_precent.append(tmp_loss_percent)
                    if len(loss_precent) >= 10:
                        if torch.mean(torch.stack(loss_precent[-5:],dim=0)) > 98.0 and opt_step > 75:
                            break
            
                    
            tmp_x_guess.detach().cpu().numpy().astype(np.float32).tofile(pf_result)
            
            ptr += args.batchsize
            

        pf_result.close()
        pf_result_lumi.close()
            
    print(which_test_thread, "done.")
