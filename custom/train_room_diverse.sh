#!/bin/bash
TRAIN_DATA=${1:-'../data_uORF/1200shape_50bg'}
TEST_DATA=${2:-'../data_uORF/room_diverse_test'}

python pl_train.py --train_dataroot $TRAIN_DATA  --test_dataroot $TRAIN_DATA \
    --n_scenes 5000 --n_img_each_scene 4 --display_grad \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 --coarse_epoch 120 \
    --no_locality_epoch 60 --z_dim 64 --num_slots 5 \
    --batch_size 1 --num_threads 10 \
# done
echo "Done"