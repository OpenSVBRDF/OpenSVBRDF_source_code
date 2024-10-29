# OpenSVBRDF: A Database of Measured Spatially-Varying Reflectance

This is the source code for our paper accepted at Siggraph Asia 2023. 

![Teaser Image](assets/sorted_teaser_new.jpg)

The database is available at [OpenSVBRDF](https://opensvbrdf.github.io/). Currently, all texture maps are available for download. Due to the large data volume, we are still looking for storage solutions for the neural representations and raw capture images. Currently, we provide the raw images of two samples for users to run this code and reproduce the results of the paper.

## Download Data & Model & Device Configurations

We offer several samples to test the results comprehensively. Each sample is about 16 GB. You can comment out some of the lines for downloading samples in the `download.sh` file based on your available storage space.

## Running Steps

We provide two approaches to run the code, you can choose any one you prefer.

### Running with Conda

We tested our code on Ubuntu 22.04.3 LTS with NVIDIA Driver Version 535.129.03, and CUDA 12.2.

1. Create a new environment and install the packages.

```
conda env create -f environment.yml
conda activate opensvbrdf
```
2. Before running each command, confirm the sample name you want to process. The output of each step will be saved in the `output/` folder.

#### Step-by-Step Instructions

First, `cd data_processing/` to set your work directory as data_processing/.

1. `cd generate_uv/` and run `run.sh` to automatically determine the final texture map area from masks and captured photos.
2. `cd parallel-patchmatch-master/` and run `run.sh` to align photos from two cameras using dense patchmatch, finding corresponding pixel positions in the secondary camera for each pixel in the primary camera.
3. `cd extract_measurements/` and run `run.sh` to extract all measurement values for each pixel in the texture map from each photo.
4. `cd finetune/` and run `run.sh` to infer the neural representations and perform the three finetune steps described in the paper. (change the `server_num` in line 21 as the number of gpus you have)
5. `cd fitting/` and run `run.sh` to fit the neural representation into 6D PBR texture maps ready for rendering.
6. `cd visualize_results` and run `run.sh` to render the neural representations and PBR texture maps, compute SSIM, and visualize MSE error against real photos for result validation.

After completing these steps, you can view the reconstructed results in the `output/render/` folder within the sample directory.

### Running with Docker 

1. Similar to the previous appoach, confirm the sample name you want to process before running each command.

2. Create docker image at the root of the project

    `docker build -t opensvbrdf .`

3. Create docker container and enter it

    `docker run -it --rm --gpus all -v ./database_data/:/app/database_data opensvbrdf`

4. Run the code, and the output will be saved in the `database_data/sample_name/output/` folder

    `bash /app/run.sh`


## License

This code is licensed under GPL-3.0. If you use our data or code in your work, please cite our paper:

```
@article{ma2023opensvbrdf,
  title={OpenSVBRDF: A Database of Measured Spatially-Varying Reflectance},
  author={Ma, Xiaohe and Xu, Xianmin and Zhang, Leyao and Zhou, Kun and Wu, Hongzhi},
  journal={ACM Transactions on Graphics (TOG)},
  volume={42},
  number={6},
  pages={1--14},
  year={2023},
  publisher={ACM New York, NY, USA}
}
```

We also acknowledge the use of code from [SIFT-Flow-GPU](https://github.com/hmorimitsu/sift-flow-gpu) for camera alignment.