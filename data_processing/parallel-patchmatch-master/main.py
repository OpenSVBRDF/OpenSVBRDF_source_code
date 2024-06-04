import cv2
import numpy as np
from patchmatch import *
import torch
import time
from sift_flow_torch import SiftFlowTorch
from third_party.flowiz import flowiz

import argparse
import cv2
import os
from datetime import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calibrate sample plane.')
    parser.add_argument('--data_root',type=str)
    parser.add_argument('--patch_size',type=int,default=3)
    parser.add_argument('--search_radius',type=int,default=50)
    parser.add_argument('--jump_radius',type=int,default=50)
    parser.add_argument('--iterations',type=int,default=10)
    
    parser.add_argument('--main_cam_id',type=int,default=0)
    parser.add_argument('--gpu_id',type=int,default=0)
    parser.add_argument('--save_root',type=str)
    args = parser.parse_args()

    save_root = args.data_root + args.save_root
    data_root = args.data_root + "sfm/"
    os.makedirs(save_root,exist_ok=True)

    time_start = datetime.now()
    patch_size = args.patch_size
    search_radius = args.search_radius
    jump_radius = args.jump_radius
    iterations = args.iterations
    
    img_ori = cv2.imread(data_root+'image0_crop.png')
    ref_ori = cv2.imread(data_root+'image1_crop.png')

    imgs = [img_ori, ref_ori]

    sift_step_size = 1
    sift_flow = SiftFlowTorch(
        cell_size=20,
        step_size=sift_step_size,
        is_boundary_included=True,
        num_bins=8,
        cuda=True,
        fp16=True,
        return_numpy=False
    )
    device = torch.device("cuda:{}".format(args.gpu_id))
    
    img = torch.from_numpy(img_ori).to(device)
    ref = torch.from_numpy(ref_ori).to(device)
    print('Warm-up step, will be slow on GPU')
    torch.cuda.synchronize()
    start = time.perf_counter()
    descs = sift_flow.extract_descriptor(imgs)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print('Time: {:.03f} ms'.format((end - start) * 1000))
        
    src_img = descs[0].unsqueeze(0)
    ref_img = descs[1].unsqueeze(0)

    height = src_img.shape[2]
    width = ref_img.shape[3]
    initial_NNF = torch.meshgrid(torch.arange(height),torch.arange(width))
    initial_NNF = torch.stack([initial_NNF[0],initial_NNF[1]],dim=-1)
    initial_NNF = initial_NNF + torch.randint(-30,30,initial_NNF.shape)

    src_h = height - patch_size + 1
    src_w = width - patch_size + 1
    initial_NNF[:,:,0]=torch.clamp(initial_NNF[:,:,0],0, src_h-1)
    initial_NNF[:,:,1]=torch.clamp(initial_NNF[:,:,1],0, src_w-1)
    initial_NNF = initial_NNF.to(device)

    pm=PatchMatch(src_img,ref_img,patch_size,initial_NNF=initial_NNF,device=device)
    nnf = pm.run(num_iters=iterations, rand_search_radius=search_radius, jump_radius=jump_radius, allow_diagonals=True)

    recon_img = pm.reconstruct_without_avg(ref.permute(2,0,1).unsqueeze(0).to(src_img.dtype))
    
    cv2.imwrite(data_root+"recon.png", recon_img)

    nnf = nnf.cpu().numpy().astype(np.int32)
    nnf.astype(np.int32).tofile(save_root+"warp.bin")
    

    
    
    
    