# -*- coding:utf-8 -*-
# @time :2023.09.12
# @author :wangfy


import argparse

parser = argparse.ArgumentParser(description="PyTorch implementation of image classification")

# ========================= Data Configs ==========================
parser.add_argument('--num_classes', type=int, default=12, help='the numbers of the image classification task')
parser.add_argument('--input_size', nargs='+', type=int)
parser.add_argument('--root_dir', default="/data/wangfy/rsna-2023-abdominal-trauma-detection/train_images_nii_croped_z_1.0_add_colon_duodenum/", type=str, help='the database root dir')
parser.add_argument('--train_db', default="/data/wangfy/rsna-2023-abdominal-trauma-detection/split_csv/overall/db/train", type=str, help='the database of training samples')
parser.add_argument('--val_db', default="/data/wangfy/rsna-2023-abdominal-trauma-detection/split_csv/overall/db/val/", type=str, help='the database of validation samples')
parser.add_argument('--test_db', default="/data/wangfy/rsna-2023-abdominal-trauma-detection/split_csv/overall/db/test/", type=str, help='the database of validation samples')
parser.add_argument('--remove_large_image', default=False, type=bool, help='whether to remove large image or not')
parser.add_argument('--size_csv', default="/data/wangfy/rsna-2023-abdominal-trauma-detection/train_images_nii_croped_z_1.0_add_colon_duodenum.csv", type=str, help='size info of the datasets')
parser.add_argument('--data_resample', default=False, action="store_true")
parser.add_argument('--data_region', default="all", type=str, help='data region')

# ========================= Data Configs for test ==========================
parser.add_argument('--batch_size_test', default=16, type=int, help='mini-batch size for test')

# ========================= Data Aug Configs ==========================

# ========================= Model Configs ==========================
parser.add_argument('--model_name', type=str, default="resnet18_3d")
parser.add_argument('--swimunetr_size', default=12, type=int,
                    metavar='N', help='swimunetr_size (default: 12)')
parser.add_argument('--init_weights', default='', type=str, help = "weight path")
parser.add_argument('--resume', default=False, action="store_true")
parser.add_argument('--single_loss', default='softmax_ce', type=str, help = "singel loss func")
parser.add_argument('--amp', default=False, action="store_true")


# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=24, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--warmup_epoch', type=int, default=5)
parser.add_argument('--lr_decay_rate', type=float, default=0.1)
parser.add_argument('--warmup_multiplier', type=int, default=100)    

parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_type', default='cosine', type=str,
                    metavar='LRtype', help='learning rate type')
parser.add_argument('--lr_steps', default=[50, 100], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')

parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')


# ========================= Monitor Configs ==========================
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')


# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')


parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--output', type=str, default='./outputs', help='save dir for logs and outputs')


parser.add_argument('--backend', default='nccl', type=str, help='Pytorch DDP backend')
parser.add_argument('--local_rank', default=-1, type=int)



