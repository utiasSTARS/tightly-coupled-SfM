#!/usr/bin/env bash
d=$(date +%Y%m%d%H%M)
python3 run_mono_training --iterations 4 --img_resolution 'med' --minibatch 8  --data_dir '/media/datasets/ScanNet-downsized'  --train_seq 'all'  --val_seq 'scene0000_00' --test_seq 'scene0000_01' --date $d --lr 1e-4 --wd 0 --num_epochs 20 --lr_decay_epoch 4 --save_results --min_depth 0.03 --max_depth 3
