for variable in 18
do
    class_name=paper
    formatted_variable=$(printf "%04d" $variable)
    data_root=../../../"$class_name""$formatted_variable"/
    
    dir_name=raw_images

    images_number=129
    cam_num=2
    main_cam_id=0
    texture_resolution=1024

    save_root=../../../$class_name"$formatted_variable"/output/texture_"$texture_resolution"/

    model_path=../../../database_model/
    config_dir=../../device_configuration/

    need_undistort=true
    color_check=true
    need_scale=true
    need_warp=true
    phto_validation=false
    down_size=2

    lighting_pattern_num=64
    line_pattern_num=64

    python extract.py $data_root $dir_name $save_root $images_number $cam_num $main_cam_id $config_dir $model_path $texture_resolution $lighting_pattern_num --line_pattern_num $line_pattern_num --down_size $down_size --need_undistort $need_undistort --color_check $color_check --need_scale $need_scale --need_warp $need_warp

done
