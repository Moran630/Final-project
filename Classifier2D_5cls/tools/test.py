import os
import random
import pandas as pd
import sys
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn import functional as F
import torch.distributed as dist
import warnings
warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 添加路径
print(BASE_DIR)
sys.path.append(BASE_DIR)

from models import ResNet18, ResNet18_rnn
from loss import MultiTaskLoss
from utils import init_logger, torch_distributed_zero_first, AverageMeter, distributed_sum, get_model_result, distributed_concat
from utils import  get_scheduler, parser
from dataset import DefaultLoaderCls5_test, collate_test, DefaultLoaderCls5Rnn_test
from common import save_csv
 


SEED = 42  # 35202
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def test(rank, local_rank, device, args):
    logger = init_logger(log_file=args.output + f'/log_test', rank=rank)
    with torch_distributed_zero_first(rank):
        if args.model_name == 'resnet18':
            test_dataset = DefaultLoaderCls5_test(args.test_db, args, logger=logger, mode='test', input_size=args.input_size)
        elif args.model_name == 'resnet18_rnn':
            test_dataset = DefaultLoaderCls5Rnn_test(args.test_db, args, logger=logger, mode='test', input_size=args.input_size)
        logger.info(f"Num test examples = {len(test_dataset)}")

    label_names = test_dataset.label_names
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, rank=rank, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size_test,
        sampler=test_sampler,
        collate_fn=collate_test,
        num_workers=args.workers, 
        pin_memory=True)

    
    label_names = test_dataset.label_names
    criterion = MultiTaskLoss(label_names, label_smoothing=0.0)

    
    if args.model_name == 'resnet18':
        model = ResNet18(num_classes=args.num_classes, num_channels=args.num_channels)
    elif args.model_name == 'resnet18_rnn':
        model = ResNet18_rnn(num_classes=args.num_classes, num_channels=args.num_channels)
    else:
        print(args.model, 'not support!')
        raise
    print(model)
    
    if args.init_weights == '':
        logger.warning(f"the checkpoint if null!")
        raise
    else:
        logger.info(f"Loading weight from {args.init_weights}")
        checkpoint = torch.load(args.init_weights, map_location=device)
        epoch = checkpoint['epoch']
        state = model.state_dict()
        state.update(checkpoint['state_dict'])
        model.load_state_dict(state, strict=True)  # , strict=False

    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    test_losses = {}
    for name in label_names:
        test_losses[name] = AverageMeter()
    test_total_loss = AverageMeter()
    model.eval()
    with torch.no_grad():
        total_img_files = []
        total_probs = []
        eval_pbar = tqdm(test_loader, desc=f'test evaluating', position=1, disable=False if rank in [-1, 0] else True)
        for step, (img, label_infos, img_files) in enumerate(eval_pbar):
            img = img.to(device)
            output = model(img)
            total_img_files.extend(img_files)
            output_probs = get_model_result(output)
            total_probs.append(output_probs)
            losses_dict, loss = criterion(output, label_infos)
            test_total_loss.update(loss.item(), img.size(0))
            for name in label_names:
                test_losses[name].update(losses_dict[name].item(), img.size(0))
            
        total_probs = torch.cat(total_probs, 0)
        
        total_img_files = np.array(total_img_files).reshape(-1, 1)

        if rank != -1:
            save_file = os.path.join(args.output, 'test_result', str(epoch), 'submission_' + str(rank) + '.csv')
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            final_columns = ['img_file', 
                             'bowel_healthy', 'bowel_injury', 
                             'extravasation_healthy', 'extravasation_injury', 
                             'kidney_healthy','kidney_low','kidney_high', 
                             'liver_healthy','liver_low','liver_high', 
                             'spleen_healthy','spleen_low','spleen_high']
            final_records = np.concatenate([total_img_files, total_probs.cpu().numpy()], axis=1)
            save_csv(save_file, final_records.tolist(), final_columns)
        
        test_total_num = torch.tensor(test_total_loss.count).to(img.device)
        test_total_loss_sum = torch.tensor(test_total_loss.sum).to(img.device)
        test_losses_sum = {}
        for name in label_names:
            test_losses_sum[name] = torch.tensor(test_losses[name].sum).to(img.device)
        
        if rank == 0:
            logger.info(f"test_total_loss = {test_total_loss.avg:.4f}")
            
            for name in label_names:
                test_loss = test_losses[name]
                logger.info(f"test_loss_{name} = {test_loss.avg:.4f}")

        if rank != -1:
            test_total_num = distributed_sum(test_total_num.clone().detach())
            test_total_loss_sum = distributed_sum(test_total_loss_sum.clone().detach())
            test_loss_avg = test_total_loss_sum / test_total_num

            for name in label_names:
                test_losses_sum[name] = distributed_sum(test_losses_sum[name].clone().detach())

            test_losses_avg = {}
            for name in label_names:
                test_losses_avg[name] = test_losses_sum[name] / test_total_num

        test_loss_avg = test_loss_avg.cpu().numpy()
        test_total_num = test_total_num.cpu().numpy()
        for name in label_names:
            test_losses_avg[name] = test_losses_avg[name].cpu().numpy()
        
        if rank == 0:
            logger.info(f"[Test all gather loss]")
            logger.info(f"test_total_loss_dist_avg = {test_loss_avg:.4f}, test_total_num = {test_total_num}" )
            for name in label_names:
                test_loss_avg_tmp = test_losses_avg[name]
                logger.info(f"test_loss_dist_avg_{name} = {test_loss_avg_tmp:.4f}")

            dfs = []
            save_file =  os.path.join(args.output, 'test_result', str(epoch), 'submission.csv')
            # merge db in rank 0
            for i in range(torch.distributed.get_world_size()):
                input_csv = os.path.join(args.output, 'test_result', str(epoch), 'submission_' + str(i) + '.csv')
                if not os.path.exists(input_csv):
                    raise
                current_df = pd.read_csv(input_csv)
                dfs.append(current_df)
            df = pd.concat(dfs, join='inner', ignore_index=True)
            df = df.drop_duplicates().reset_index(drop=True)
            df.to_csv(save_file, index=False)
            logger.info(f"Saving combined file: {save_file}")

            # 利用df计算series维度指标




def distributed_init(backend="gloo", port=None):

    num_gpus = torch.cuda.device_count()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )


if __name__ == '__main__':

    args = parser.parse_args()
    args.input_size = tuple(args.input_size)
    print(args)
    distributed_init(backend = args.backend)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    args.world_size = int(os.environ["WORLD_SIZE"])

    print(f"[init] == local rank: {local_rank}, global rank: {rank} == devices: {device}")

    test(rank, local_rank, device, args)