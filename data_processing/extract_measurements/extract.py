'''
This is the experimental code for paper "Ma et al., OpenSVBRDF: A Database of Measured Spatially-Varying Reflectance, SIGGRAPH Asia 2023".
This script is suboptimal and experimental.
There may be redundant lines and functionalities.

Xiaohe Ma, 2024/02
'''

'''
This script extracts measurement data from images by traversing 64 lighting patterns, 64 line patterns, and 1 transparency image.
'''


import numpy as np
import cv2
import argparse
import os
import struct
import sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

sys.path.append("../camera_related/")
from camera_config import Camera
from datetime import datetime


def str2bool(v):
  # Convert string to boolean
  return v.lower() in ("yes", "true", "t", "1")

def get_bool_type(parser):
    # Register the boolean type for argparse
    parser.register('type','bool',str2bool)

def warp_image(img,H,shift,h0=424,w0=1728,need_shift=True):
    """
    Warp the image using the homography matrix and shift values.

    Parameters:
    img (ndarray): Input image.
    H (ndarray): Homography matrix.
    shift (ndarray): Shift values for warping.
    h0 (int): Starting height index.
    w0 (int): Starting width index.
    need_shift (bool): Flag to apply shifting.

    Returns:
    ndarray: Warped image.
    """
    height,width = img.shape[:2]

    # Warp the image using the homography matrix
    img_warp = cv2.warpPerspective(img, H, (width,height),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    if not need_shift:
        return img_warp

    # Create an expanded shift map
    shift_expand = np.zeros([height,width,2],np.int32)
    shift_expand[h0:h0+shift.shape[0],w0:w0+shift.shape[1],:] = shift
    shift_expand[h0:h0+shift.shape[0],w0:w0+shift.shape[1],0] += h0
    shift_expand[h0:h0+shift.shape[0],w0:w0+shift.shape[1],1] += w0
    
    # Transpose and apply the shift
    img_warp = np.transpose(img_warp,(2,0,1))
    temp = np.zeros_like(img_warp)
    dest_i=shift_expand[:, :, 0] 
    dest_j=shift_expand[:, :, 1]
    mesh=tuple([dest_i,dest_j])
    temp[0,:]= img_warp[0][mesh]
    temp[1,:]= img_warp[1][mesh]
    temp[2,:]= img_warp[2][mesh]
    
    return np.transpose(temp,(1,2,0))

def error_warning():
    print("ERROR OCCUR!!")
    exit()
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    get_bool_type(parser)
    parser.add_argument("data_root",default="F:/Turbot_freshmeat/test0/")
    parser.add_argument("dir_name",default="raw_images/")
    parser.add_argument("save_root",type=str)
    parser.add_argument("images_number",type=int,default=129)
    parser.add_argument("cam_num",type=int,default=2)
    parser.add_argument("main_cam_id",type=int,default=0)
    parser.add_argument('config_dir',type=str)
    parser.add_argument("model_path",type=str)
    parser.add_argument("texture_resolution",type=int)

    parser.add_argument("lighting_pattern_num",type=int)
    parser.add_argument("--line_pattern_num",type=int,default=64)
    parser.add_argument("--translucent_pattern_num",type=int,default=1)

    parser.add_argument("--down_size",type=int,default=2)
    parser.add_argument("--need_undistort",type="bool")
    parser.add_argument("--color_check",type="bool")
    parser.add_argument("--need_scale",type="bool")
    parser.add_argument("--need_warp",type="bool")
    
    args = parser.parse_args()
    
    time_start = datetime.now()

    cameras = [Camera(args.config_dir+f"intrinsic{which_cam}.yml", args.config_dir+f"extrinsic{which_cam}.yml") for which_cam in range(args.cam_num)]

    img_height = cameras[0].get_height()
    img_width = cameras[0].get_width()

    img_down_height = img_height // args.down_size
    img_down_width = img_width // args.down_size
    
    image_root = args.data_root + args.dir_name + "/"
    save_root = args.save_root
    
    if args.need_warp:
        img = cv2.imread(args.data_root+"sfm/image0_crop.png")
        H = np.fromfile(save_root+"H.bin",np.float64).reshape([3,3])
        shift = np.fromfile(save_root+"warp.bin",np.int32).reshape(img.shape[0], img.shape[1], 2)

    roi = np.fromfile(save_root+f"roi_{args.down_size}.bin",np.int32).reshape([6,])
    print(roi)
    if args.need_scale:
        with open(args.model_path+"maxs.bin","rb") as pf:
            maxs = np.fromfile(pf,np.float32).reshape([args.lighting_pattern_num,1])
            
    plane_scalar_2cams = np.fromfile(args.config_dir+"plane_scalar.bin",np.float32).reshape([2, img_height, img_width])
    
    mask_name = f"mask_udt_cam{args.main_cam_id:02d}_{args.down_size}.exr" 
    
    mask = cv2.imread(image_root+mask_name, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)

    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]
    mask = np.repeat(mask[:, :, :1], 3, axis=2)

    valid_idxes = np.where(mask == 1.0) 
    valid_num = valid_idxes[0].shape[0] // 3
    with open(image_root+f"cam0{args.main_cam_id}_index_nocc.bin","wb") as pf:
        pf.write(struct.pack("i",valid_num))
        valid_idxes_2d = np.stack(valid_idxes[:2], axis=1)[::3]
        valid_idxes_2d.astype(np.int32).tofile(pf)
        
    tmp_map = {(valid_idxes_2d[i][1],valid_idxes_2d[i][0]) : i for i in range(valid_idxes_2d.shape[0])}
    
    tmp_cam_uvs = np.fromfile(save_root+f"uvs_cam{args.main_cam_id}.bin", np.float32).reshape([-1,2])
    tmp_cam_uvs = tmp_cam_uvs.astype(np.int32)
    cam_tex_uvs = tuple(tmp_cam_uvs)
    texel_num = len(cam_tex_uvs)
    print("pixel number: ", texel_num)
    texel_sequence = np.arange(texel_num)
    
    m_len = 3
    pf_map = {}
    
    plane_scalar = np.split(plane_scalar_2cams, 2, axis=0)
    plane_scalar = [np.expand_dims(np.squeeze(plane_scalar[i],axis=0),axis=-1) for i in range(args.cam_num)]
    
    color_mat = [np.fromfile(args.config_dir + f"color_mat{which_cam}.bin", np.float32).reshape([3,3]) for which_cam in range(args.cam_num)]
    
    for which_image in range(args.lighting_pattern_num+args.line_pattern_num+args.translucent_pattern_num):
        if which_image == 0:
            pf_map["measurements"] = open(save_root+f"measurements_{args.texture_resolution}.bin","wb")
            cam_num = 2
            collector = np.zeros([texel_num, cam_num, args.lighting_pattern_num, m_len],np.float32)

        elif which_image == args.lighting_pattern_num:
            
            pf_map["line_measurements"] = open(save_root+f"line_measurements_{args.texture_resolution}.bin","wb")
            cam_num = 1
            collector = np.zeros([texel_num, cam_num, args.line_pattern_num, m_len],np.float32)
            
        elif which_image == args.lighting_pattern_num+args.line_pattern_num:
            pf_map["translucent_measurements"] = open(save_root+"translucent_measurements.bin","wb")
            cam_num = 1
            collector = np.zeros([texel_num, cam_num, args.translucent_pattern_num, m_len],np.float32)

        need_shift = True if which_image < args.lighting_pattern_num else False
        
        for which_cam in range(cam_num):
            try:
                tmp_image = cv2.imread(image_root+f"img{which_image:0>5d}_cam{which_cam:02d}.exr", cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)[:,:,::-1]
            except:
                error_warning()

            tmp_image = tmp_image * plane_scalar[which_cam]
            tmp_image = cameras[which_cam].undistort_2steps(tmp_image)
            
            tmp_image = cv2.GaussianBlur(tmp_image,(roi[4],1),0)
            tmp_image = cv2.GaussianBlur(tmp_image,(1,roi[5]),0)
            tmp_image = cv2.resize(tmp_image,(tmp_image.shape[1]//args.down_size,tmp_image.shape[0]//args.down_size), cv2.INTER_LINEAR)
            
            tmp_image = np.maximum(np.matmul(tmp_image,color_mat[which_cam]), 0) if args.color_check else tmp_image
                
            if args.need_warp and which_cam != args.main_cam_id:
                tmp_image = warp_image(tmp_image,H,shift,roi[1],roi[0],need_shift)
            
            tmp_measurements = np.transpose(tmp_image[valid_idxes].reshape((-1,3,1)),[0,2,1])

            if which_image < args.lighting_pattern_num and args.need_scale:
                tmp_measurements *= maxs[which_image]

            print(image_root+f"img_udt{which_image:0>5d}_cam{which_cam:02d}.exr")
                
            if which_image < args.lighting_pattern_num:
                for which_tex in range(texel_num):
                    collector[which_tex,which_cam,which_image,:] = tmp_measurements[tmp_map[tuple(cam_tex_uvs[which_tex])]] 
                
                if which_image == args.lighting_pattern_num-1 and which_cam == args.cam_num-1:
                    collector[:,0] *= 187.363 / 2.0
                    collector[:,1] *= 169.155 / 2.0
                    print("measurements file : ", collector.shape)
                    collector.astype(np.float32).tofile(pf_map["measurements"])
                    pf_map["measurements"].close()
                    
                    
            elif which_image < args.lighting_pattern_num + args.line_pattern_num:
                which_pattern = which_image - args.lighting_pattern_num
                for which_tex in range(texel_num):
                    collector[which_tex,which_cam,which_pattern,:] = tmp_measurements[tmp_map[tuple(cam_tex_uvs[which_tex])]] 
                if which_image == args.lighting_pattern_num+args.line_pattern_num-1 and which_cam == cam_num-1:
                    collector[:,0] *= 187.363 / 2.0
                    print("line pattern mesurements file : ", collector.shape)
                    collector.astype(np.float32).tofile(pf_map["line_measurements"])
                    pf_map["line_measurements"].close()
                    

            elif which_image < args.lighting_pattern_num + args.line_pattern_num + args.translucent_pattern_num:
                which_pattern = which_image - args.lighting_pattern_num - args.line_pattern_num
                for which_tex in range(texel_num):
                    collector[which_tex,which_cam,which_pattern,:] = tmp_measurements[tmp_map[tuple(cam_tex_uvs[which_tex])]] 
                if which_image == args.lighting_pattern_num+args.line_pattern_num + args.translucent_pattern_num-1 and which_cam == cam_num-1:
                    collector[:,0] *= 187.363 / 2.0
                    
                    print("transparent pattern mesurements file : ", collector.shape)
                    collector.astype(np.float32).tofile(pf_map["translucent_measurements"])
                    pf_map["translucent_measurements"].close()
                    
                    break
               
    cam_num = 1
    
    translucent_measurements = np.fromfile(save_root+"translucent_measurements.bin",np.float32).reshape([texel_num, cam_num, args.translucent_pattern_num, m_len]) 
    empty_measurements = np.fromfile(args.config_dir+"empty_measurements.bin",np.float32).reshape([texel_num, cam_num, args.translucent_pattern_num, m_len])

    alpha = translucent_measurements / (empty_measurements+1e-6)
    alpha = np.mean(alpha, axis=-1, keepdims=True)
    alpha = np.repeat(alpha, 3, axis=-1)
    alpha = np.where(alpha > 1.0, np.ones_like(alpha), alpha)
    
    alpha = alpha.reshape([args.texture_resolution,args.texture_resolution,args.translucent_pattern_num, m_len])
    alpha = np.transpose(alpha,(2,0,1,3))

    for which_img in range(args.translucent_pattern_num):
        cv2.imwrite(save_root+f"translucent_{args.lighting_pattern_num+args.line_pattern_num+which_img}.exr", alpha[which_img])
