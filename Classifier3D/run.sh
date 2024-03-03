# unset LD_LIBRARY_PATH
#train
OUTPUT_PATH=./outputs_aug_multi_head_noweight_withoutanyhead_and_withinjuryloss_resample_weight
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -u -m torch.distributed.launch --nproc_per_node 4 --master_port 29224 ./tools/train.py \
    --model_name resnet18_3d \
    --lr  0.0001 --epochs 100  --batch-size 6  -j 8 \
    --output $OUTPUT_PATH \
    --input_size 128 128 128 \
    --num_classes 11 \
    --optimizer Adam \
    --train_db /data/wangfy/rsna-2023-abdominal-trauma-detection/split_csv/overall/db/train/ \
    --val_db /data/wangfy/rsna-2023-abdominal-trauma-detection/split_csv/overall/db/val/ \
    --data_region all \
    # --data_resample


