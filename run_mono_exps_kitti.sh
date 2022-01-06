#!/usr/bin/env bash
d=$(date +%Y%m%d%H%M)

### only train w 00-08 for odometry results
python3 run_mono_training.py --iterations 4 --img_resolution 'med' --minibatch 6 --data_dir '/media/datasets/KITTI-odometry-downsized-stereo'  --train_seq '00_02' '02_02' '05_02' '06_02' '07_02' '08_02' '00_03' '02_03' '05_03' '06_03' '07_03' '08_03' --val_seq '09_02' --test_seq '10_02' --date $d --lr 1e-4 --wd 0 --num_epochs 20 --lr_decay_epoch 7 --save_results --min_depth 0.06 --max_depth 2.67


### Train on Eigen split (for depth eval)
#python3 run_mono_training.py --iterations 4 --img_resolution 'med' --minibatch 6 --data_format 'eigen' --data_dir '/path/to/KITTI-eigen-preprocessed' --date $d --lr 1e-4 --wd 0 --num_epochs 15 --lr_decay_epoch 5 --save_results --min_depth 0.1 --max_depth 2.67

