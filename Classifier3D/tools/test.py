import os
import random
import time
import timm
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

from models import ResNet18_3d, ResNet18_3d_single, SWINUNETR_CLS
from loss import MultiTaskLoss_withoutany, SingleTaskLoss, SingleTaskLossSigmoid
from utils import init_logger, torch_distributed_zero_first, AverageMeter, distributed_sum, get_model_result, distributed_concat
from utils import  get_scheduler, parser
from dataset import DefaultLoader, collate_test
from common import save_csv


SEED = 42  # 35202
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def test(rank, local_rank, device, args):
    logger = init_logger(log_file=args.output + f'/log_test', rank=rank)
    with torch_distributed_zero_first(rank):
        test_dataset = DefaultLoader(args.test_db, args, logger=logger, mode='test', input_size=args.input_size)
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

    
    data_region = args.data_region
    if data_region == 'all':
        criterion = MultiTaskLoss_withoutany(label_names)
    else:
        if args.single_loss == 'bce':
            criterion = SingleTaskLossSigmoid(label_names)
        else:
            criterion = SingleTaskLoss(label_names, mode='test')

    
    if args.model_name == 'resnet18_3d':
        if args.data_region == 'all':
            model = ResNet18_3d(num_classes=args.num_classes)
        else:
            model = ResNet18_3d_single(num_classes=args.num_classes)
    elif args.model_name == 'SWINUNETR':
        model = SWINUNETR_CLS(
            in_channels=1,
            num_classes=args.num_classes,
            feature_size=args.swimunetr_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.3,
            use_checkpoint=False,
            )
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
        total_uids = []
        total_probs = []
        eval_pbar = tqdm(test_loader, desc=f'test evaluating', position=1, disable=False if rank in [-1, 0] else True)
        for step, (img, label_infos, uids) in enumerate(eval_pbar):
            img = img.to(device)
            output = model(img)
            uids = uids.to(device)
            total_uids.append(uids)
            output_probs = get_model_result(output, data_region)
            total_probs.append(output_probs)
            losses_dict, loss = criterion(output, label_infos)
            test_total_loss.update(loss.item(), img.size(0))
            for name in label_names:
                test_losses[name].update(losses_dict[name].item(), img.size(0))
        total_uids = torch.cat(total_uids, 0)
        total_probs = torch.cat(total_probs, 0)
        
        if rank != -1:
            total_uids_dist = distributed_concat(total_uids)
            total_probs_dist = distributed_concat(total_probs)
        total_dist_cat = torch.cat([total_uids_dist.unsqueeze(1), total_probs_dist], dim=1)
        total_dist_cat = total_dist_cat.cpu().numpy()

        if rank == 0:
            save_file = os.path.join(args.output, 'test_result', str(epoch), 'submission.csv')
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            if data_region == 'all':
                final_columns = ['series_id','bowel_healthy','bowel_injury','extravasation_healthy','extravasation_injury','kidney_healthy','kidney_low','kidney_high','liver_healthy','liver_low','liver_high','spleen_healthy','spleen_low','spleen_high']
            else:
                final_columns = ['series_id', data_region + '_healthy', data_region + '_low', data_region + '_high']
            save_csv(save_file, total_dist_cat.tolist(), final_columns)
        
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
    # args.input_size = tuple(args.input_size)
    print(args)
    distributed_init(backend = args.backend)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    args.world_size = int(os.environ["WORLD_SIZE"])

    print(f"[init] == local rank: {local_rank}, global rank: {rank} == devices: {device}")

    test(rank, local_rank, device, args)