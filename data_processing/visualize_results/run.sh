
for variable in 18
do
    class_name=paper
    formatted_variable=$(printf "%04d" $variable)
    data_root=../../database_data/$class_name$formatted_variable/output/
    # data_root=../../../$class_name$variable/output/
    save_root=$data_root/render/
    
    model_root=../../database_model
    pattern_file=$model_root/opt_W_64.bin
    model_file=$model_root/latent_48_24_500000_2437925.pkl
    tex_resolution=1024

    gpu_id=1
    main_cam_id=0
    shape_latent_len=48
    color_latent_len=8
    lighting_pattern_num=64

    
    python rendering.py $data_root $save_root $model_file $pattern_file --tex_resolution $tex_resolution --main_cam_id $main_cam_id --lighting_pattern_num $lighting_pattern_num

done
