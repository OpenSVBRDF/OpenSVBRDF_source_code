import os
import numpy as np
import argparse
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import sys
sys.path.append("../camera_related/")
from camera_config import Camera

def undistort_masks(cameras, ldr_path, hdr_path, down_size):
    """
    Undistort the mask and image files using the provided camera parameters.

    Parameters:
    cameras (list): List of Camera objects for each camera.
    ldr_path (str): Path to the LDR images.
    hdr_path (str): Path to the HDR images.
    down_size (int): Factor by which to downsize the images.
    """
    for which_cam in range(len(cameras)):
        # Read and undistort LDR images
        mask = cv2.imread(ldr_path + f"mask_cam{which_cam:02d}.png")
        mask = cameras[which_cam].undistort_2steps(mask)
        cv2.imwrite(ldr_path + f"mask_udt_cam{which_cam:02d}.png", mask)

        mask = cv2.resize(mask, (mask.shape[1]//down_size, mask.shape[0]//down_size), cv2.INTER_NEAREST)
        cv2.imwrite(ldr_path + f"mask_udt_cam{which_cam:02d}_{down_size}.png", mask)
        
        img = cv2.imread(ldr_path + f"{which_cam}_0.png")
        img = cameras[which_cam].undistort_2steps(img)
        cv2.imwrite(ldr_path + f"{which_cam}_0_udt.png", img)

        # Undistort and save HDR mask for the main camera (which_cam == 0)
        if which_cam == 0:
            mask = cv2.imread(hdr_path + f"mask_cam{which_cam:02d}.exr", 6)
            mask = cameras[which_cam].undistort_2steps(mask)
            cv2.imwrite(hdr_path + f"mask_udt_cam{which_cam:02d}.png",np.clip(mask*255, 0, 255).astype(np.uint8))
            cv2.imwrite(hdr_path + f"mask_udt_cam{which_cam:02d}.exr", mask)
            mask = cv2.resize(mask, (mask.shape[1]//down_size, mask.shape[0]//down_size), cv2.INTER_NEAREST)
            cv2.imwrite(hdr_path + f"mask_udt_cam{which_cam:02d}_{down_size}.png", np.clip(mask*255, 0, 255).astype(np.uint8))
            cv2.imwrite(hdr_path + f"mask_udt_cam{which_cam:02d}_{down_size}.exr", mask)
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract measurements.')
    parser.add_argument('data_root', type=str)
    parser.add_argument('save_root', type=str)
    parser.add_argument('config_dir', type=str, help="Path to camera parameters.")
    parser.add_argument('--cam_num', type=int, default=2)
    parser.add_argument('--main_cam_id', type=int, default=0)
    parser.add_argument('--texture_resolution', type=int, default=1024)
    parser.add_argument('--down_size',type=int, default=2)
    
    args = parser.parse_args()
    
    # Define paths for LDR and HDR images
    ldr_path = args.data_root + "sfm/"
    hdr_path = args.data_root + "raw_images/"
    output_root = args.data_root + args.save_root + f"texture_{args.texture_resolution}/"
    
    os.makedirs(output_root, exist_ok=True)

    # Load camera configurations
    cameras = [Camera(args.config_dir + f"intrinsic{which_cam}.yml", args.config_dir + f"extrinsic{which_cam}.yml") for which_cam in range(args.cam_num)]
    
    undistort_masks(cameras, ldr_path, hdr_path, args.down_size)
    

    height = cameras[0].get_height()
    width = cameras[0].get_width()

    positions = np.fromfile(args.config_dir + f"positions_{args.down_size}.bin", np.float32).reshape([height//args.down_size, width//args.down_size,3])

    mask = cv2.imread(hdr_path + f"mask_udt_cam{args.main_cam_id:02d}_{args.down_size}.exr", 6)
    camera = cameras[args.main_cam_id]

    # Find valid indices where the mask is valid
    valid_idxes = np.where(mask[:,:,0] == 1.0)
    # Extract valid positions
    valid_positions = positions[valid_idxes[1],valid_idxes[0]].reshape([-1,3])
    
    # Project valid positions to image coordinates and downsize
    uv = camera.project(valid_positions.T) / args.down_size
    uv = np.maximum(uv,0)
    uv[:,0] = np.minimum(uv[:,0],width//args.down_size-1)
    uv[:,1] = np.minimum(uv[:,1],height//args.down_size-1)
   
    # Determine bounding box of valid positions
    x_min, x_max = np.min(valid_positions[:, 0]), np.max(valid_positions[:, 0])
    y_min, y_max = np.min(valid_positions[:, 1]), np.max(valid_positions[:, 1])
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Scale and expand bounding box for safe region
    scale = 0.065
    expand = 50 // args.down_size
    z = -122

    x_start = int(x_min + scale * x_range)
    x_end = int(x_max - scale * x_range)
    y_start = int(y_min + scale * y_range)
    y_end = int(y_max - scale * y_range)
    length = np.minimum(x_end - x_start, y_end - y_start)
    
    uvs = []
    # Project corners of bounding box to image coordinates
    for step_x in range(2):
        for step_y in range(2):
            x = x_start + step_x * length 
            y = y_start + step_y * length
            
            uv = camera.project(np.array([x,y,z]).reshape([3,1])) / args.down_size
            uvs.append(uv)

    # Calculate minimum and maximum UV coordinates and apply expansion
    uvs = np.concatenate(uvs, axis=0)
    uv_min = np.min(uvs,axis=0).astype(np.int32) - expand
    uv_max = np.max(uvs,axis=0).astype(np.int32) + expand
    
    # Generate grid of positions for texture mapping
    x = np.linspace(x_start, x_end, args.texture_resolution)
    y = np.linspace(y_start, y_end, args.texture_resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z)
    positions = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    positions.astype(np.float32).tofile(output_root+"positions.bin")
    
    # Generate UV coordinates for texture mapping   
    u = np.linspace(0, args.texture_resolution - 1, args.texture_resolution)
    v = np.linspace(0, args.texture_resolution - 1, args.texture_resolution)
    U, V = np.meshgrid(u, v)
    uvs = np.stack([U, V], axis=-1).reshape(-1, 2)
    uvs.astype(np.int32).tofile(output_root+"texturemap_uv.bin")

    for which_cam in range(args.cam_num):
        # Project positions to UV coordinates for each camera
        uv = cameras[which_cam].project(positions.T) / args.down_size
        uv.astype(np.float32).tofile(output_root+f"uvs_cam{args.main_cam_id}.bin")

        # Compute region of interest (ROI) for the main camera
        if which_cam == 0:

            uv = uv.reshape([args.texture_resolution, args.texture_resolution, 2]).astype(np.int32)
            
            u_delta = np.max(np.abs(uv[:,1:,0] - uv[:,:args.texture_resolution-1,0])) 
            v_delta = np.max(np.abs(uv[1:,:,1] - uv[:args.texture_resolution-1,:,1])) 

            roi = np.array([uv_min[0], uv_min[1], uv_max[0], uv_max[1], int(np.ceil(u_delta/2+1))*2-1, int(np.ceil(v_delta/2+1))*2-1])
            
            roi.astype(np.int32).tofile(output_root+"roi_{}.bin".format(args.down_size))

    

    
    