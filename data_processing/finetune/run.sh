for variable in 18
do
    class_name=paper
    formatted_variable=$(printf "%04d" $variable)
    name=$class_name$formatted_variable
    tex_resolution=1024

    data_root=../../database_data/$name/output/
    model_root=../../database_model/
    model_file_name=model_state_450000.pkl

    lighting_pattern_num=32
    measurement_len=1
    shape_latent_len=48
    color_latent_len=8
    python infer.py $data_root $model_root $model_file_name $lighting_pattern_num $measurement_len $tex_resolution --layers 6 --shape_latent_len $shape_latent_len --color_latent_len $color_latent_len 
    
    if_dump=1
    save_lumi=1
    thread_num=4
    server_num=4
    which_server=0
    lighting_pattern_num=64
    main_cam_id=0
    model_file_name=$model_root/latent_48_24_500000_2437925.pkl

    pattern_file_name=$model_root/opt_W_64.bin
    
    finetune_use_num=64
    step="123"
    if_continue=0

    python split_data_and_prepare_for_server.py  $data_root $lighting_pattern_num $thread_num $server_num $which_server $tex_resolution $main_cam_id $shape_latent_len $color_latent_len

    save_root=$data_root/latent/
    data_root=$data_root/data_for_server/

    python fitting_master.py $data_root $thread_num $server_num $which_server $if_dump $lighting_pattern_num $finetune_use_num $tex_resolution $main_cam_id $model_file_name $pattern_file_name $shape_latent_len $color_latent_len $save_lumi $step --if_continue $if_continue

    python gather_finetune_results.py $data_root $save_root $thread_num $server_num $tex_resolution

done