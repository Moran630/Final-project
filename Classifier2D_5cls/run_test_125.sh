# shuffle train
OUTPUT_PATH=./outputs_cls5_rnn_timestep16_slide_reverse_shuffle_125
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -u -m torch.distributed.launch --nproc_per_node 4 --master_port 29225 ./tools/test.py \
    --model_name resnet18_rnn \
    --batch_size_test 64  -j 4 \
    --output $OUTPUT_PATH \
    --input_size 224 224 \
    --num_classes 13 \
    --num_channels 1 \
    --root_dir /data/wangfy/rsna-2023-abdominal-trauma-detection/train_images \
    --init_weights ./outputs_cls5_rnn_timestep16_slide_reverse_shuffle_125/models/epoch_91.pth \
    --test_db /data/wangfy/rsna-2023-abdominal-trauma-detection/split_csv/overall/db/test/ \
    --image_level_labels_csv /data/wangfy/rsna-2023-abdominal-trauma-detection/image_level_labels.csv \
    --organ_info_csv /data/wangfy/rsna-2023-abdominal-trauma-detection/organ_instance.csv \
    --time_step 16 \
    --reverse