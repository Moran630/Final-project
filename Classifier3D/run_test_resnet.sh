# unset LD_LIBRARY_PATH
#test
OUTPUT_PATH=./outputs_resnet18
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -u -m torch.distributed.launch --nproc_per_node 4 --master_port 29225 ./tools/test.py \
    --model_name resnet18_3d \
    --input_size 144 144 144 \
    --batch_size_test 8  -j 4 \
    --output $OUTPUT_PATH \
    --num_classes 11 \
    --init_weights ./outputs_resnet18/models/best.pth \
    --root_dir /data/wangfy/rsna-2023-abdominal-trauma-detection/ \
    --test_db /data/wangfy/rsna-2023-abdominal-trauma-detection/split_csv/overall/db/test/ \
    --data_region all \
    --single_loss mutliloss


