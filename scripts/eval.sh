#!/bin/bash
DATAROOT=${1:-'../data_uORF/room_diverse_test'}
CHECKPOINT=${2:-'./'}

python test.py --train_dataroot "" --test_dataroot $DATAROOT --n_scenes 500 --n_img_each_scene 4 \
    --checkpoint $CHECKPOINT \
    --load_size 128 --input_size 128 --render_size 8 --frustum_size 128 --bottom \
    --n_samp 256 --z_dim 64 --num_slots 5 --gpus 1 \

echo "Done"