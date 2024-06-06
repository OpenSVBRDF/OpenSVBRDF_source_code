'''
 This is the experimental code for paper ``Ma et al., OpenSVBRDF: A Database of Measured Spatially-Varying Reflectance, SIGGRAPH Asia 2023``.
 This script is suboptimal and experimental.
 There may be redundant lines and functionalities.

 Xiaohe Ma, 2024/02
'''


'''
 This script gathers finetuning results and organizes documents for further operation.

 by Leyao
'''
import numpy as np
import cv2
import os
import OpenEXR, array
import Imath
from sklearn.decomposition import PCA
import argparse

FLOAT = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="fitting using multi card")
    
    parser.add_argument("data_root")
    parser.add_argument("save_root")
    parser.add_argument("thread_num",type=int)
    parser.add_argument("server_num",type=int)
    parser.add_argument("tex_resolution",type=int)
    parser.add_argument("--latent_len",type=int,default=72)

    args = parser.parse_args()
    os.makedirs(args.save_root, exist_ok=True)

    pf_latent = {}
    for latent_id in range(4):
        pf_latent[f"{latent_id}"] = open(args.save_root+f"pass{latent_id}_latent_{args.tex_resolution}.bin","wb")
    
    for latent_id in range(4):
        for which_thread in range(args.thread_num*args.server_num):
            tmp_latent = np.fromfile(args.data_root+f"{which_thread}/pass{latent_id}_latent_{args.tex_resolution}.bin", np.float32)
            tmp_latent.astype(np.float32).tofile(pf_latent[f"{latent_id}"])
            
    for latent_id in range(4):
        pf_latent[f"{latent_id}"].close()

    data = np.fromfile(args.save_root+f"pass3_latent_{args.tex_resolution}.bin",np.float32).reshape([args.tex_resolution,args.tex_resolution,-1])
    
    print(data.shape)

    header = OpenEXR.Header(args.tex_resolution,args.tex_resolution)

    channels_num = data.shape[2]
    new_channels = {}

    for which_channel in range(data.shape[2]):
        tmpdata = data[:,:,which_channel].reshape([args.tex_resolution,args.tex_resolution,1]).tobytes()
        new_channels['L_{}'.format(which_channel)] = tmpdata

    header['channels'] = dict([(c, FLOAT) for c in new_channels.keys()])
    header['compression'] = Imath.Compression(Imath.Compression.PIZ_COMPRESSION)
    out = OpenEXR.OutputFile(args.save_root+"latent.exr", header)
    out.writePixels(new_channels)

    data = data.reshape([-1, args.latent_len])
    pca = PCA(n_components=3)
    pca.fit(data)
    data_reduction = pca.transform(data)
    data_reduction_min = np.min(data_reduction)
    data_reduction_max = np.max(data_reduction)
    data_reduction_delta = data_reduction_max - data_reduction_min
    data_reduction = (data_reduction - data_reduction_min) / data_reduction_delta
    data_reduction = data_reduction.reshape([args.tex_resolution,args.tex_resolution,3])
    
    cv2.imwrite(args.save_root+"latent_pca.png",data_reduction*255)

        