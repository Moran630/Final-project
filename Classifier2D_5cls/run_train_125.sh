OUTPUT_PATH=./outputs_cls5_rnn_timestep16_slide_reverse_125_step20
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -u -m torch.distributed.launch --nproc_per_node 4 --master_port 29224 ./tools/train.py \
    --model_name resnet18_rnn \
    --lr  0.001 --epochs 100  --batch-size 12 -j 6 \
    --output $OUTPUT_PATH \
    --input_size 224 224 \
    --num_classes 13 \
    --num_channels 1 \
    --optimizer Adam \
    --root_dir /data/wangfy/rsna-2023-abdominal-trauma-detection/train_images \
    --train_db /data/wangfy/rsna-2023-abdominal-trauma-detection/split_csv/overall/db/train/ \
    --val_db /data/wangfy/rsna-2023-abdominal-trauma-detection/split_csv/overall/db/val/ \
    --image_level_labels_csv /data/wangfy/rsna-2023-abdominal-trauma-detection/image_level_labels.csv \
    --organ_info_csv /data/wangfy/rsna-2023-abdominal-trauma-detection/organ_instance.csv \
    --time_step 20
    # --data_resample \

