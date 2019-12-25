CUDA_VISIBLE_DEVICES=1,2,3 \
python train_fashion_original.py \
    --img_size 128 \
    --shortcut_layers 1 \
    --inject_layers 1 \
    --experiment_name exp_2 \
    --experiment_dir exp_2