# unset LD_LIBRARY_PATH
#train
OUTPUT_PATH=./outputs_resnet18
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -u -m torch.distributed.launch --nproc_per_node 4 --master_port 29224 ./tools/train_single.py \
    --model_name resnet18_3d \
    --lr  0.01 --epochs 100  --batch-size 2  -j 8 \
    --output $OUTPUT_PATH \
    --input_size 144 144 144 \
    --num_classes 11 \
    --optimizer SGD \
    --momentum 0.9 \
    --root_dir /data/wangfy/rsna-2023-abdominal-trauma-detection/ \
    --train_db /data/wangfy/rsna-2023-abdominal-trauma-detection/split_csv/overall/db/train/ \
    --val_db /data/wangfy/rsna-2023-abdominal-trauma-detection/split_csv/overall/db/val/ \
    --data_region all \
    --single_loss mutliloss \


