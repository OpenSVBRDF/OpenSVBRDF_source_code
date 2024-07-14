'''
 This is the experimental code for paper ``Ma et al., OpenSVBRDF: A Database of Measured Spatially-Varying Reflectance, SIGGRAPH Asia 2023``.
 This script is suboptimal and experimental.
 There may be redundant lines and functionalities.

 Xiaohe Ma, 2024/02
'''


'''
 This script reads in line measurements file and interpolates the pixels by lerp(). Then, it crops the result image if needed.

 by Leyao
'''
import numpy as np
import argparse
import torch
import sys
import math
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import csv
import matplotlib.pyplot as plt
import matplotlib
sys.path.append("../finetune/")
import AUTO_planar_scanner_net_inference

TORCH_RENDER_PATH = "../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
# import torch_render
from torch_render import TorchRender
from setup_config import SetupConfig

from skimage.metrics import structural_similarity as ssim
import skimage

RENDER_SCALAR = 3e3/math.pi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root",default="")
    parser.add_argument("save_root",default="")
    parser.add_argument("model_file",type=str,default="")
    parser.add_argument("pattern_file",type=str,default="")
    parser.add_argument("--lighting_pattern_num",type=int,default=64)
    parser.add_argument("--tex_resolution",type=int,default=1024)
    parser.add_argument("--main_cam_id",type=int,default=0)
    parser.add_argument("--config_dir",type=str,default="../device_configuration/")
    parser.add_argument("--gpu_id",type=int,default=0)
    parser.add_argument("--batch_size",type=int,default=100)
    parser.add_argument("--shape_latent_len",type=int,default=48)
    parser.add_argument("--color_latent_len",type=int,default=8)
    args = parser.parse_args()

    root = args.data_root
    os.makedirs(args.save_root, exist_ok=True)
    
    compute_device = torch.device(f"cuda:{args.gpu_id}")
    setup = SetupConfig(TORCH_RENDER_PATH+"wallet_of_torch_renderer/lightstage/")
    torch_render = TorchRender(setup)

     
    m_len_perview = 3
    scalar = 255*0.04
    uvs = np.fromfile(args.data_root+"texture_1024/texturemap_uv.bin", np.int32).reshape([-1,2])

    uv_map = np.fromfile(args.data_root+f"texture_{args.tex_resolution}/uvs_cam{args.main_cam_id}.bin", np.float32).astype(np.int32)

    uv_map = uv_map.reshape([-1,2])
    uv_map = uv_map // 2
    min_uv = np.min(uv_map, axis=0)
    uv_map = uv_map - min_uv
    
    w, h = np.max(uv_map, axis=0) + 1
    

    ############# GT
    gt_result = np.fromfile(args.data_root+f"texture_{args.tex_resolution}/line_measurements_{args.tex_resolution}.bin",np.float32).reshape((-1,args.lighting_pattern_num,m_len_perview))


    gt_imgs = np.zeros([h, w, args.lighting_pattern_num, 3], dtype=np.float32)
    gt_imgs[uv_map[:, 1],  uv_map[:, 0]] = gt_result
    gt_imgs = np.transpose(gt_imgs,(2,0,1,3)) * scalar


    ############# LATENT
    train_configs = {}
    train_configs["training_device"] = 0
    train_configs["lumitexel_length"] = 64*64*3
    train_configs["shape_latent_len"] = args.shape_latent_len
    train_configs["color_latent_len"] = args.color_latent_len
    
    pretrained_dict = torch.load(args.model_file, map_location=compute_device)
    
    inference_net = AUTO_planar_scanner_net_inference.PlanarScannerNet(train_configs)
    print("loading trained model...")
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
    
    
    lighting_patterns = np.fromfile(args.pattern_file,np.float32).reshape((args.lighting_pattern_num,-1,3)).astype(np.float32)
    lighting_patterns = lighting_patterns.reshape([1,args.lighting_pattern_num,-1,3])
    lighting_patterns = torch.from_numpy(lighting_patterns).to(compute_device)

    latent_len = args.color_latent_len + args.shape_latent_len

    batch_size = args.batch_size
    lumitexel_length = 64*64*3
    all_latent_len = args.shape_latent_len + 3 * args.color_latent_len
    
    pf_latent = open(args.data_root+f"latent/pass3_latent_{args.tex_resolution}.bin","rb")

    pf_latent.seek(0,2)
    texel_num = pf_latent.tell()//all_latent_len//4
    pf_latent.seek(0,0)
    print("texel num : ", texel_num)

    texel_sequence = np.arange(texel_num)
    
    ptr = 0

    latent_result = np.zeros([texel_num,args.lighting_pattern_num,3],np.float32)
    latent_result = torch.from_numpy(latent_result).to(compute_device)

    while True:
        
        tmp_sequence = texel_sequence[ptr:ptr+batch_size]
        if tmp_sequence.shape[0] == 0:
            break
        tmp_seq_size = tmp_sequence.shape[0]
        
        tmp_latent = np.fromfile(pf_latent,np.float32,count=all_latent_len*tmp_seq_size).reshape([tmp_seq_size,all_latent_len])
        tmp_latent = torch.from_numpy(tmp_latent).to(compute_device)
        
        color_latent = tmp_latent[:,:3 * args.color_latent_len].reshape([tmp_seq_size,3,args.color_latent_len])
        shape_latent = tmp_latent[:,3 * args.color_latent_len:].reshape([tmp_seq_size,1,args.shape_latent_len]).repeat(1,3,1)
        
        color_shape_latent = torch.cat([color_latent,shape_latent],dim=-1)
        color_shape_latent = color_shape_latent.reshape([tmp_seq_size*3,latent_len])
        
        _,tmp_nn_lumi = inference_net(color_shape_latent,input_is_latent=True)
        tmp_nn_lumi = torch.max(torch.zeros_like(tmp_nn_lumi),tmp_nn_lumi)
        
        tmp_nn_lumi = tmp_nn_lumi.reshape([tmp_seq_size,3,-1]).permute(0,2,1)
        
        if ptr % 20000 == 0:
            lumi_img = torch_render.visualize_lumi(tmp_nn_lumi).cpu().numpy()
            for i in range(1):
                cv2.imwrite(args.save_root+f"{ptr+i}_latent.png",lumi_img[i,:,:,::-1]*255)
        
        tmp_measurements = torch.sum(lighting_patterns*tmp_nn_lumi.unsqueeze(dim=1),dim=2).reshape([tmp_seq_size,-1,3])
        latent_result[ptr:ptr+batch_size,:,:] = tmp_measurements

        ptr += batch_size


    latent_imgs = np.zeros([h, w, args.lighting_pattern_num, 3], dtype=np.float32)
    latent_imgs[uv_map[:, 1],  uv_map[:, 0]] = latent_result.cpu().numpy()

    latent_imgs = np.transpose(latent_imgs,(2,0,1,3)) * scalar

    ############# TEXTURE_MAPS
    texure_root = args.data_root+"texture_maps/"
    positions = cv2.imread(texure_root + "pos_texture.exr", 6)[:,:,::-1].reshape([-1,3])
    fitted_axay = cv2.imread(texure_root+"ax_ay_texture.exr",6)[:,:,::-1][:,:,:2].reshape([-1,2])
    fitted_normal = cv2.imread(texure_root+"normal_texture.exr",6)[:,:,::-1].reshape([-1,3])
    fitted_pd = cv2.imread(texure_root+"pd_texture.exr",6)[:,:,::-1].reshape([-1,3])
    fitted_ps = cv2.imread(texure_root+"ps_texture.exr",6)[:,:,::-1].reshape([-1,3])
    fitted_tangent = cv2.imread(texure_root+"tangent_texture.exr",6)[:,:,::-1].reshape([-1,3])
    
    fitted_normal = (fitted_normal - 0.5) * 2.0
    fitted_tangent = (fitted_tangent - 0.5) * 2.0

    fitted_params_rgb = np.concatenate([np.zeros([texel_num,3],np.float32), fitted_axay, fitted_pd, fitted_ps],axis=-1)
    
    texel_sequence = np.arange(texel_num)
    ptr = 0
    batch_size = args.batch_size
    
    tex_result = np.zeros([texel_num,args.lighting_pattern_num,3],np.float32)
    tex_result = torch.from_numpy(tex_result).to(compute_device)

    while True:
        
        tmp_sequence = texel_sequence[ptr:ptr+batch_size]
        if tmp_sequence.shape[0] == 0:
            break
        tmp_seq_size = tmp_sequence.shape[0]
        pos = torch.from_numpy(positions[tmp_sequence]).to(compute_device).to(torch.float32)

        params = fitted_params_rgb[tmp_sequence]
        
        params = torch.from_numpy(params).to(compute_device).to(torch.float32)
        
        n = torch.from_numpy(fitted_normal[tmp_sequence]).to(compute_device)
        t = torch.from_numpy(fitted_tangent[tmp_sequence]).to(compute_device)
        b = torch.cross(n,t)
        
        rotate_theta = torch.zeros(tmp_seq_size,1,device=compute_device,dtype=torch.float32)
        shading_frame = [n,t,b]

        lumi, end_points = torch_render.generate_lumitexel(
            params,
            pos,
            global_custom_frame=[n,t,b],
            use_custom_frame="ntb",
            pd_ps_wanted="both",
        )

        lumi = lumi.reshape(tmp_seq_size,setup.get_light_num(),3)*RENDER_SCALAR
        
        
        measurements = torch.sum(lighting_patterns*lumi.unsqueeze(dim=1),dim=2).reshape([tmp_seq_size,-1,3])
            
        tex_result[ptr:ptr+batch_size,:,:] = measurements
        ptr += batch_size
    

    tex_imgs = np.zeros([h, w, args.lighting_pattern_num, 3], dtype=np.float32)
    tex_imgs[uv_map[:, 1],  uv_map[:, 0]] = tex_result.cpu().numpy()


    tex_imgs = np.transpose(tex_imgs,(2,0,1,3)) * scalar

    
    img_num = args.lighting_pattern_num
    loss = np.zeros([img_num, 2], np.float32)
    imgs = []
    
    for which_img in range(img_num):
        gt_img = np.array(gt_imgs[which_img], np.float32)
        latent_img = np.array(latent_imgs[which_img], np.float32)
        tex_img = np.array(tex_imgs[which_img], np.float32)
        
        loss[which_img, 0] = ssim(gt_img, latent_img, data_range=255, channel_axis=-1)
        loss[which_img, 1] = ssim(gt_img, tex_img, data_range=255, channel_axis=-1)
        
        latent_error = np.abs(gt_img-latent_img).reshape([-1,3])
        tex_error = np.abs(gt_img-tex_img).reshape([-1,3])

        h,w = gt_img.shape[:2]

        gt_error = matplotlib.cm.jet(np.zeros_like(latent_error)/255)[:,0,:3]*255/2
        gt_error = gt_error.reshape([h,w,3])

        latent_error = matplotlib.cm.jet(latent_error/255)[:,0,:3]*255/2
        latent_error = latent_error.reshape([h,w,3])

        tex_error = matplotlib.cm.jet(tex_error/255)[:,0,:3]*255/2
        tex_error = tex_error.reshape([h,w,3])
        
        latent_img = cv2.putText(latent_img, '%.2f' % loss[which_img, 0], (0,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),5)
        tex_img = cv2.putText(tex_img, '%.2f' % loss[which_img, 1], (0,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),5)

        photo_img = np.concatenate([gt_img, latent_img, tex_img],axis=1)
        error_img = np.concatenate([gt_error,latent_error,tex_error],axis=1)
        img = np.concatenate([photo_img, error_img],axis=0)
        img = img[:,:,::-1]

        cv2.imwrite(args.save_root+f"{which_img}.png",img)


    np.savetxt(args.save_root + "ssim_loss.csv", loss, delimiter=',', fmt='%.2f')





        






