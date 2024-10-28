# OpenSVBRDF: A Database of Measured Spatially-Varying Reflectance

This is the source code for our paper accepted at Siggraph Asia 2023. 

![Teaser Image](assets/sorted_teaser_new.jpg)

The database is available at [OpenSVBRDF](https://opensvbrdf.github.io/). Currently, all texture maps are available for download. Due to the large data volume, we are still looking for storage solutions for the neural representations and raw capture images. Currently, we provide the raw images of two samples for users to run this code and reproduce the results of the paper.

## Data

We offer an anisotropic satin sample and an isotropic greeting card sample composed of various materials to test the results comprehensively. Download [paper0018](https://drive.google.com/file/d/1PBhkDUvGb9goTIzc_9HxCAtOB72KGw9K/view?usp=sharing) and [satin0002](https://drive.google.com/file/d/1tKWYCvvX073X8HIgc2kjC8_QoGUM5QVP/view?usp=sharing) and place the data folders in the `database_data` folder.

## Model

Download the [model](https://drive.google.com/file/d/1px3Ij1B7GIESWhAAm0-MHOhwR6yjWaVB/view?usp=drive_link) and place its contents in the `database_model/` folder.

## Device Configuration

Download the [device configuration files](https://drive.google.com/file/d/1dIqEQcImBUaTGfsy0SVb8S6Pjua5u317/view?usp=drive_link) and place its contents in the `data_processing/` folder as `device_configuration/`.

## Running Steps

We provide two approaches to run the code, you can choose any one you prefer.

### Running with Conda

1. Create a new environment and install the packages.

    `conda env create -f environment.yml`
2. Before running each command, confirm the sample name you want to process. The output of each step will be saved in the `output/` folder.

#### Step-by-Step Instructions

1. `cd generate_uv/` and run `run.sh` to automatically determine the final texture map area from masks and captured photos.
2. `cd parallel-patchmatch-master/` and run `run.sh` to align photos from two cameras using dense patchmatch, finding corresponding pixel positions in the secondary camera for each pixel in the primary camera.
3. `cd extract_measurements/` and run `run.sh` to extract all measurement values for each pixel in the texture map from each photo.
4. `cd finetune/` and run `run.sh` to infer the neural representations and perform the three finetune steps described in the paper.
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