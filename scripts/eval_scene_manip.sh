#!/bin/bash
DATAROOT=${1:-'room_chair_test'}
CHECKPOINT=${2:-'./'}
PORT=8077
python -m visdom.server -p $PORT &>/dev/null &
python test.py --dataroot $DATAROOT --n_scenes 500 --n_img_each_scene 4 \
    --checkpoints_dir $CHECKPOINT --name room_chair_models --exp_id latest --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 64 --render_size 8 --frustum_size 128 \
    --n_samp 256 --z_dim 64 --num_slots 5 \
    --model 'uorf_manip' --dataset_mode 'multiscenes_manip'
echo "Done"
