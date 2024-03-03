# unset LD_LIBRARY_PATH
#test
OUTPUT_PATH=./outputs_aug_multi_head_noweight_withoutanyhead_and_withinjuryloss
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -u -m torch.distributed.launch --nproc_per_node 4 --master_port 29225 ./tools/test.py \
    --model_name resnet18_3d \
    --batch_size_test 16  -j 4 \
    --output $OUTPUT_PATH \
    --num_classes 11 \
    --init_weights /data/wangfy/github/kaggle/RSNA/code/Classifier/outputs_aug_multi_head_noweight_withoutanyhead_and_withinjuryloss/models/epoch_9.pth \
    --test_db /data/wangfy/rsna-2023-abdominal-trauma-detection/split_csv/overall/db/val/ \
    --without_any_injury \


