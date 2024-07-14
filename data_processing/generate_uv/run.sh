for variable in 18
do
    root=../../database_data/
    class_name=paper

    formatted_variable=$(printf "%04d" $variable)
    data_root=$root$class_name$formatted_variable/

    save_root=output/
    config_dir=../device_configuration/

    cam_num=2
    main_cam_id=0
    texture_resolution=1024
    down_size=2

    python extract_plane.py $data_root $save_root $config_dir --cam_num $cam_num --main_cam_id $main_cam_id --texture_resolution $texture_resolution --down_size $down_size 

    python warp.py $data_root $save_root --main_cam_id $main_cam_id --texture_resolution $texture_resolution --down_size $down_size 

done
