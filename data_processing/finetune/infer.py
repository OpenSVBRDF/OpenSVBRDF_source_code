'''
 This is the experimental code for paper ``Ma et al., OpenSVBRDF: A Database of Measured Spatially-Varying Reflectance, SIGGRAPH Asia 2023``.
 This script is suboptimal and experimental.
 There may be redundant lines and functionalities.

 Xiaohe Ma, 2024/02
'''

'''
This script infers input data by loading PlanarScannerNet using a pre-trained neural network model.
'''

import torch
import argparse
import random
import sys
import numpy as np
import os

import planar_scanner_net_inference 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data_root")
    parser.add_argument("model_root")
    parser.add_argument("model_file_name")
    parser.add_argument("lighting_pattern_num",type=int)
    parser.add_argument("m_len",type=int)
    parser.add_argument("tex_resolution",type=int)
    parser.add_argument("--training_gpu",type=int,default=0)
    parser.add_argument("--batch_size",type=int,default=100)
    parser.add_argument("--cam_num",type=int,default=2)
    parser.add_argument("--main_cam_id",type=int,default=0)

    parser.add_argument("--layers",type=int,default=8)
    parser.add_argument("--shape_latent_len",type=int,default=2)
    parser.add_argument("--color_latent_len",type=int,default=2)
    parser.add_argument("--need_dump",action="store_true")


    args = parser.parse_args()
    
    latent_len = args.shape_latent_len + args.color_latent_len
     
    # Define configuration parameters
    train_configs = {}
    train_configs["rendering_devices"] = [torch.device("cuda:{}".format(args.training_gpu))] # for multiple GPU
    train_configs["training_device"] = torch.device("cuda:{}".format(args.training_gpu))
    train_configs["layers"] = args.layers
    train_configs["lighting_pattern_num"] = args.lighting_pattern_num
    train_configs["m_len"] = args.m_len
    train_configs["train_lighting_pattern"] = False
    train_configs["lumitexel_length"] = 64*64*3
    train_configs["cam_num"] = args.cam_num
    train_configs["main_cam_id"] = args.main_cam_id
    train_configs["latent_len"] = latent_len
    train_configs["color_latent_len"] = args.color_latent_len
    train_configs["shape_latent_len"] = args.shape_latent_len
    train_configs["data_root"] = args.data_root
    train_configs["batch_size"] = args.batch_size*3
    train_configs["pre_load_buffer_size"] = 500000

    save_root = args.data_root + f"texture_{args.tex_resolution}/"
    
    os.makedirs(save_root,exist_ok=True)
    
    # Load the pre-trained model
    model = planar_scanner_net_inference.PlanarScannerNet(train_configs)
    inference_device = train_configs["training_device"]
    pretrained_dict = torch.load(args.model_root + args.model_file_name, map_location='cuda:0')
    
    print("loading trained model...")
    something_not_found = False
    model_dict = model.state_dict()
    for k,_ in model_dict.items():
        if k not in pretrained_dict:
            print("not found:", k)
            something_not_found = True
    if something_not_found:
        exit()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
    model.to(inference_device)
    
    
    model.eval()
    pf_map = {}
    pf_map["latent"] = open(save_root+f"latent_{args.tex_resolution}.bin","wb")
    
    record_size_byte = args.cam_num * args.m_len * args.lighting_pattern_num * 2 * 4
    
    if args.m_len == 1:
        record_size_byte *= 3
    pf_measurements = open(args.data_root+f"texture_{args.tex_resolution}/measurements_{args.tex_resolution}.bin", "rb")
    pf_measurements.seek(0,2)
    texel_num = pf_measurements.tell()//record_size_byte
    pf_measurements.seek(0,0)
    print("texel num : ", texel_num)
    texel_sequence = np.arange(texel_num)
    start_ptr = 0
    ptr = start_ptr
    batch_size = args.batch_size
    lumitexel_length = train_configs["lumitexel_length"]
    lighting_pattern_num = args.lighting_pattern_num 

    # Process data in batches
    while True:
        tmp_sequence = texel_sequence[ptr:ptr+batch_size]
        if tmp_sequence.shape[0] == 0:
            break
        tmp_seq_size = tmp_sequence.shape[0]

        # Read raw measurements
        tmp_measurements_raw = np.fromfile(pf_measurements,np.float32,count=record_size_byte//4*tmp_seq_size).reshape([tmp_seq_size,args.cam_num,lighting_pattern_num*2,3])
        tmp_measurements_raw = torch.from_numpy(tmp_measurements_raw).to(inference_device)

        tmp_measurements_raw_mean = torch.mean(tmp_measurements_raw,dim=-1)
        
        tmp_measurements_raw_mean = tmp_measurements_raw_mean[:,:,::2] - tmp_measurements_raw_mean[:,:,1::2] 
        
        # Perform inference
        res = model(tmp_measurements_raw_mean)
        latent = res["nn_latent"] 

        latent.astype(np.float32).tofile(pf_map["latent"])

   
        ptr += batch_size

    for a_key in pf_map:
        pf_map[a_key].close()
