'''
 This is the experimental code for paper ``Ma et al., OpenSVBRDF: A Database of Measured Spatially-Varying Reflectance, SIGGRAPH Asia 2023``.
 This script is suboptimal and experimental.
 There may be redundant lines and functionalities.

 Xiaohe Ma, 2024/02
'''


'''
 This script splits the data and prepares for the server.

'''

import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(usage="split relighting brdf not slice")

parser.add_argument("data_root")
parser.add_argument("lighting_pattern_num",type=int)
parser.add_argument("thread_num",type=int)
parser.add_argument("server_num",type=int)
parser.add_argument("which_server",type=int)
parser.add_argument("tex_resolution",type=int)
parser.add_argument("main_cam_id",type=int)
parser.add_argument("shape_latent_len",type=int)
parser.add_argument("color_latent_len",type=int)

args = parser.parse_args()

if __name__ == "__main__":
    data_root = args.data_root + f"texture_{args.tex_resolution}/"
    target_root = args.data_root + "data_for_server/"
  
    latent_len = args.shape_latent_len + args.color_latent_len
    
    m_len_perview = 3
    total_thread_num = args.thread_num*args.server_num

    os.makedirs(target_root, exist_ok=True)

    pf_latent = open(data_root+f"latent_{args.tex_resolution}.bin")
    pf_latent.seek(0,2)
    pixel_num = pf_latent.tell() //4//latent_len
    print("[SPLITTER]pixel num:",pixel_num)
    pf_latent.seek(0,0)

    pf_pos = open(data_root+"positions.bin")

    pf_measurement = open(data_root+f"line_measurements_{args.tex_resolution}.bin")
    
    num_per_thread = int(pixel_num//total_thread_num)
    
    ptr = 0
    for thread_id in range(total_thread_num):
        tmp_dir = target_root+f"{thread_id}/"
        os.makedirs(tmp_dir, exist_ok=True)

        cur_batchsize = num_per_thread if (not thread_id == total_thread_num-1) else (pixel_num-(total_thread_num-1)*num_per_thread)

        tmp_latents = np.fromfile(pf_latent,np.float32,cur_batchsize*latent_len)
        tmp_latents.astype(np.float32).tofile(tmp_dir+f"latent_{args.tex_resolution}.bin")
        tmp_positions = np.fromfile(pf_pos,np.float32,cur_batchsize*3)
        tmp_positions.astype(np.float32).tofile(tmp_dir+"positions.bin")

        
        tmp_measurements = np.fromfile(pf_measurement,np.float32, cur_batchsize*args.lighting_pattern_num*m_len_perview)
        tmp_measurements.astype(np.float32).tofile(tmp_dir+f"gt_measurements_{args.tex_resolution}.bin")
        print("thread:",thread_id," num:",cur_batchsize)

        ptr += cur_batchsize

    remain_data = np.fromfile(pf_measurement,np.uint8)
    
    if len(remain_data) > 0:
        print("meaurements file is not at the end!")
        exit()
    
    pf_latent.close()
    pf_pos.close()
    pf_measurement.close()


    
    