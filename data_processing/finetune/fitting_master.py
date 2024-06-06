'''
 This is the experimental code for paper ``Ma et al., OpenSVBRDF: A Database of Measured Spatially-Varying Reflectance, SIGGRAPH Asia 2023``.
 This script is suboptimal and experimental.
 There may be redundant lines and functionalities.

 Xiaohe Ma, 2024/02
'''

'''
 This script runs the three steps of the per-pixel fine-tuning process.

'''

import os
from subprocess import Popen
import argparse
import queue
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import warnings
warnings.filterwarnings('ignore')

exclude_gpu_list = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="fitting using multi cards")
    
    parser.add_argument("data_root")
    parser.add_argument("thread_num",type=int)
    parser.add_argument("server_num",type=int)
    parser.add_argument("which_server",type=int)
    parser.add_argument("if_dump",type=int)
    parser.add_argument("lighting_pattern_num",type=int)
    parser.add_argument("finetune_use_num",type=int)
    parser.add_argument("tex_resolution",type=int)
    parser.add_argument("main_cam_id",type=int)
    parser.add_argument("model_file",type=str)
    parser.add_argument("pattern_file",type=str)
    parser.add_argument("shape_latent_len",type=int)
    parser.add_argument("color_latent_len",type=int)

    parser.add_argument("save_lumi",type=int)
    parser.add_argument("step",type=str)
    parser.add_argument("--if_continue",type=int,default=0)

    args = parser.parse_args()

    deviceIDs = [0,1,2,3]
    gpu_num = len(deviceIDs)
    print("available gpu num:",gpu_num)
    pool = []
    p_log_f = open(args.data_root+"fitting_log_all.txt","w",buffering=1)
    ##################################
    # step 1 
    ##################################
    if '1' in args.step: 
        thread_per_gpu = [0]*len(deviceIDs)
        q = queue.Queue()
        [q.put(i) for i in range(0, args.server_num*args.thread_num)]
        try_num = [0]*(args.thread_num*args.server_num)
        pool = []
        while not q.empty():
            which_thread = q.get()
            try_num[which_thread] = try_num[which_thread]+1
            print("starting thread:{}".format(which_thread))
            which_gpu = which_thread%gpu_num
            my_env = os.environ.copy()

            this_args = [
                    "python",
                    "finetune_pass1.py",
                    "--data_for_server_root",
                    "{}".format(args.data_root),
                    "--lighting_pattern_num",
                    "{}".format(args.lighting_pattern_num),
                    "--finetune_use_num",
                    "{}".format(args.finetune_use_num),
                    "--thread_ids",
                    "{}".format(which_thread),
                    "--gpu_id",
                    "{}".format(deviceIDs[which_gpu]),
                    "--tex_resolution",
                    "{}".format(args.tex_resolution),
                    "--total_thread_num",
                    "{}".format(args.thread_num*args.server_num),
                    "--main_cam_id",
                    "{}".format(args.main_cam_id),
                    "--model_file",
                    "{}".format(args.model_file),
                    "--pattern_file",
                    "{}".format(args.pattern_file),
                    "--shape_latent_len",
                    "{}".format(args.shape_latent_len),
                    "--color_latent_len",
                    "{}".format(args.color_latent_len)
            ]
            if args.if_dump:
                this_args.append("--need_dump")
            if args.save_lumi:
                this_args.append("--save_lumi")
            theProcess = Popen(
                this_args,
                env=my_env
            )
            pool.append(theProcess)
            thread_per_gpu[which_gpu] += 1 
            if thread_per_gpu.count(3) == len(thread_per_gpu):
                exit_codes = [p.wait() for p in pool]
                print("fitting rhod rhos exit codes:",exit_codes)
                for i in range(len(exit_codes)):
                    crash_thread_id = int(pool[i].args[9])
                    if exit_codes[i] != 0 and try_num[crash_thread_id] < 2:
                        print("thread:{} crashed!".format(crash_thread_id))
                        q.put(crash_thread_id)
                pool = []
                thread_per_gpu = [0]*len(deviceIDs)
        exit_codes = [ p.wait() for p in pool ]
        print("fitting rhod rhos exit codes:",exit_codes)
        p_log_f.write("fitting rhod rhos exit codes:{}\n".format(exit_codes))

    ##################################
    #step 2 
    ##################################
    if '2' in args.step: 
        thread_per_gpu = [0]*len(deviceIDs)
        q = queue.Queue()
        [q.put(i) for i in range(0, args.server_num*args.thread_num)]

        try_num = [0]*(args.thread_num*args.server_num)
        pool = []
        while not q.empty():
            which_thread = q.get()
            try_num[which_thread] = try_num[which_thread]+1
            print("starting thread:{}".format(which_thread))
            which_gpu = which_thread%gpu_num
            my_env = os.environ.copy()

            this_args = [
                    "python",
                    "finetune_pass2.py",
                    "--data_for_server_root",
                    "{}".format(args.data_root),
                    "--lighting_pattern_num",
                    "{}".format(args.lighting_pattern_num),
                    "--finetune_use_num",
                    "{}".format(args.finetune_use_num),
                    "--thread_ids",
                    "{}".format(which_thread),
                    "--gpu_id",
                    "{}".format(deviceIDs[which_gpu]),
                    "--tex_resolution",
                    "{}".format(args.tex_resolution),
                    "--total_thread_num",
                    "{}".format(args.thread_num*args.server_num),
                    "--main_cam_id",
                    "{}".format(args.main_cam_id),
                    "--model_file",
                    "{}".format(args.model_file),
                    "--pattern_file",
                    "{}".format(args.pattern_file),
                    "--shape_latent_len",
                    "{}".format(args.shape_latent_len),
                    "--color_latent_len",
                    "{}".format(args.color_latent_len)
            ]
            if args.if_dump:
                this_args.append("--need_dump")
            if args.save_lumi:
                this_args.append("--save_lumi")
            theProcess = Popen(
                this_args,
                env=my_env
            )

            pool.append(theProcess)
            thread_per_gpu[which_gpu] += 1 
            if thread_per_gpu.count(3) == len(thread_per_gpu):
                exit_codes = [p.wait() for p in pool]
                print("fitting rhod rhos exit codes:",exit_codes)
                for i in range(len(exit_codes)):
                    crash_thread_id = int(pool[i].args[9])
                    if exit_codes[i] != 0 and try_num[crash_thread_id] < 2:
                        print("thread:{} crashed!".format(crash_thread_id))
                        q.put(crash_thread_id)
                pool = []
                thread_per_gpu = [0]*len(deviceIDs)
        exit_codes = [ p.wait() for p in pool ]
        print("fitting rhod rhos exit codes:",exit_codes)
        p_log_f.write("fitting rhod rhos exit codes:{}\n".format(exit_codes))

    ##################################
    #step 3 
    ##################################
    if '3' in args.step:
        thread_per_gpu = [0]*len(deviceIDs)
        q = queue.Queue()
        [q.put(i) for i in range(0, args.server_num*args.thread_num)]

        try_num = [0]*(args.thread_num*args.server_num)
        pool = []
        while not q.empty():
            which_thread = q.get()
            try_num[which_thread] = try_num[which_thread]+1
            print("starting thread:{}".format(which_thread))
            which_gpu = which_thread%gpu_num
            my_env = os.environ.copy()

            this_args = [
                    "python",
                    "finetune_pass3.py",
                    "--data_for_server_root",
                    "{}".format(args.data_root),
                    "--lighting_pattern_num",
                    "{}".format(args.lighting_pattern_num),
                    "--finetune_use_num",
                    "{}".format(args.finetune_use_num),
                    "--thread_ids",
                    "{}".format(which_thread),
                    "--gpu_id",
                    "{}".format(deviceIDs[which_gpu]),
                    "--tex_resolution",
                    "{}".format(args.tex_resolution),
                    "--total_thread_num",
                    "{}".format(args.thread_num*args.server_num),
                    "--main_cam_id",
                    "{}".format(args.main_cam_id),
                    "--model_file",
                    "{}".format(args.model_file),
                    "--pattern_file",
                    "{}".format(args.pattern_file),
                    "--shape_latent_len",
                    "{}".format(args.shape_latent_len),
                    "--color_latent_len",
                    "{}".format(args.color_latent_len),
                    "--if_continue",
                    "{}".format(args.if_continue)
            ]
            if args.if_dump:
                this_args.append("--need_dump")
            if args.save_lumi:
                this_args.append("--save_lumi")
            theProcess = Popen(
                this_args,
                env=my_env
            )

            pool.append(theProcess)
            thread_per_gpu[which_gpu] += 1 
            if thread_per_gpu.count(2) == len(thread_per_gpu):
                exit_codes = [p.wait() for p in pool]
                print("fitting rhod rhos exit codes:",exit_codes)
                for i in range(len(exit_codes)):
                    crash_thread_id = int(pool[i].args[9])
                    if exit_codes[i] != 0 and try_num[crash_thread_id] < 2:
                        print("thread:{} crashed!".format(crash_thread_id))
                        q.put(crash_thread_id)
                pool = []
                thread_per_gpu = [0]*len(deviceIDs)
        exit_codes = [ p.wait() for p in pool ]
        print("fitting rhod rhos exit codes:",exit_codes)
        p_log_f.write("fitting rhod rhos exit codes:{}\n".format(exit_codes))
