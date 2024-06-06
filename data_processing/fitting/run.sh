for variable in 18
do
    class_name=paper
    formatted_variable=$(printf "%04d" $variable)
    name=$class_name$formatted_variable
    tex_resolution=1024

    data_root=../../../$name/output/
    save_root=../../../$name/output/texture_maps/
    config_dir=../torch_renderer/wallet_of_torch_renderer/lightstage/
    model_file=../../../database_model/latent_48_24_500000_2437925.pkl
    train_device="cuda:0"
    python fit_latent.py $class_name $data_root $save_root --train_device $train_device --config_dir $config_dir --model_file $model_file

done
