# unset LD_LIBRARY_PATH
#test
OUTPUT_PATH=./outputs_SWIMUNTER_all
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -u -m torch.distributed.launch --nproc_per_node 4 --master_port 29224 ./tools/test.py \
    --model_name SWINUNETR \
    --batch_size_test 4  -j 4 \
    --output $OUTPUT_PATH \
    --input_size 144 144 144 \
    --num_classes 11 \
    --init_weights ./outputs_SWIMUNTER_all/models/best.pth \
    --root_dir /data/wangfy/rsna-2023-abdominal-trauma-detection/ \
    --test_db /data/wangfy/rsna-2023-abdominal-trauma-detection/split_csv/overall/db/test/ \
    --data_region all \
    --single_loss mutliloss