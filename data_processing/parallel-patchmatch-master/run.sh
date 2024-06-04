
for variable in 18
do
    class_name=paper

    formatted_variable=$(printf "%04d" $variable)
    data_root=../../../database_data/"$class_name""$formatted_variable"/

    texture_resolution=1024
    save_root=output/texture_"$texture_resolution"/

    main_cam_id=0
    patch_size=7
    search_radius=50
    jump_radius=20
    iterations=10

    gpu_id=0

    python main.py --data_root $data_root --patch_size $patch_size --search_radius $search_radius --jump_radius $jump_radius --iterations $iterations --main_cam_id $main_cam_id --gpu_id $gpu_id --save_root $save_root

done