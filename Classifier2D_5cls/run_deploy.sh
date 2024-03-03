OUTPUT_PATH=./outputs_cls5_rnn_timestep16_slide/
CUDA_VISIBLE_DEVICES=0,1 \
python -u ./tools/deploy.py \
    --model_name resnet18_rnn \
    --output $OUTPUT_PATH \
    --num_classes 13 \
    --input_size 224 224 \
    --num_channels 1 \
    --root_dir /data/wangfy/rsna-2023-abdominal-trauma-detection/train_images \
    --init_weights /fileser51/wangfy/github/kaggle/RSNA/code/Classifier2D_5cls/outputs_cls5_rnn_timestep16_slide/models/epoch_22.pth \
    --test_db /data/wangfy/rsna-2023-abdominal-trauma-detection/split_csv/overall/db/test/ \
    --image_level_labels_csv /data/wangfy/rsna-2023-abdominal-trauma-detection/image_level_labels.csv \
    --organ_info_csv /data/wangfy/rsna-2023-abdominal-trauma-detection/organ_instance.csv \
    --time_step 16 \
    --output_pt ./outputs_cls5_rnn_timestep16_slide/epoch_22_gpu0.pt