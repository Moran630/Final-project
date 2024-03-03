# unset LD_LIBRARY_PATH
#train
OUTPUT_PATH=./outputs_SWIMUNTER_all
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -u -m torch.distributed.launch --nproc_per_node 4 --master_port 29224 ./tools/train_single.py \
    --model_name SWINUNETR \
    --lr  0.0001 --epochs 200  --batch-size 2  -j 8 \
    --output $OUTPUT_PATH \
    --input_size 144 144 144 \
    --num_classes 11 \
    --optimizer AdamW \
    --momentum 0.9 \
    --resume \
    --init_weights ./pretrain_weights/swin_unetr.tiny_5000ep_f12_lr2e-4_pretrained.pt \
    --root_dir /data/wangfy/rsna-2023-abdominal-trauma-detection/ \
    --train_db /data/wangfy/rsna-2023-abdominal-trauma-detection/split_csv/overall/db/train/ \
    --val_db /data/wangfy/rsna-2023-abdominal-trauma-detection/split_csv/overall/db/val/ \
    --data_region all \
    --single_loss mutliloss
