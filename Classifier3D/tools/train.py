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
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
print(BASE_DIR)
sys.path.append(BASE_DIR)

from models import ResNet18_3d, SWINUNETR_CLS
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from loss import MultiTaskLoss_withoutany, MultiTaskLoss_withany
from utils import init_logger, torch_distributed_zero_first, AverageMeter, distributed_sum
from utils import  get_scheduler, parser
from dataset import DefaultLoader, collate


SEED = 42  # 35202
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def train(rank, local_rank, device, args):
    check_rootfolders(args)
    logger = init_logger(log_file=args.output + f'/log', rank=rank)

    with torch_distributed_zero_first(rank):
        train_dataset = DefaultLoader(args.train_db, args, logger=logger, mode='train', input_size=args.input_size)
        val_dataset = DefaultLoader(args.val_db, args, logger=logger, mode='val', input_size=args.input_size)
        # badcase_dataset = DefaultLoader(args.badcase_db, args, logger=logger, mode='badcase', input_size=(args.input_size, args.input_size))

        logger.info(f"Num train examples = {len(train_dataset)}")
        logger.info(f"Num val examples = {len(val_dataset)}")
        # logger.info(f'Num Badcase exampless = {len(badcase_dataset)}')

    label_names = train_dataset.label_names
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, rank=rank, shuffle=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        collate_fn=collate,
        num_workers=args.workers, 
        pin_memory=True)
    


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=rank,shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate,
        num_workers=args.workers, 
        pin_memory=True,
        drop_last=False)

    
    criterion = MultiTaskLoss_withoutany(label_names)

    if args.model_name == 'resnet18_3d':
        model = ResNet18_3d(num_classes=args.num_classes)
    elif args.model_name == 'SWINUNETR':
        model = SWINUNETR_CLS(
            in_channels=1,
            num_classes=11,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.2,
            use_checkpoint=False,
            )
    else:
        print(args.model, 'not support!')
        raise
    # print(model)

    val_min_loss = 1e9
    best_epoch = 0

    start_epoch = args.start_epoch
    if args.resume and args.init_weights != '':
        if args.model_name == 'SWINUNETR':
            model_dict = torch.load(args.init_weights,map_location=device)
            state_dict = model_dict["state_dict"]
            # fix potential differences in state dict keys from pre-training to
            # fine-tuning
            if "module." in list(state_dict.keys())[0]:
                logger.info("Tag 'module.' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("module.", "")] = state_dict.pop(key)
            if "swin_vit" in list(state_dict.keys())[0]:
                logger.info("Tag 'swin_vit' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
            # We now load model weights, setting param `strict` to False, i.e.:
            # this load the encoder weights (Swin-ViT, SSL pre-trained), but leaves
            # the decoder weights untouched (CNN UNet decoder).
            model.load_state_dict(state_dict, strict=False)
            logger.info("Using pretrained self-supervised Swin UNETR backbone weights !")
            pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info("Total parameters count", pytorch_total_params)
        else:
            logger.info(f"Loading weight from {args.init_weights}")
            checkpoint = torch.load(args.init_weights,map_location=device)
            epoch = checkpoint['epoch']
            state = model.state_dict()
            state.update(checkpoint['state_dict'])
            model.load_state_dict(state)  # , strict=False
            # optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = epoch + 1

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    
    # optim_param_groups[0].update({'lr': args.lr * 0.1})
    # optim_param_groups[1].update({'lr': args.lr})
    optimizer = getattr(torch.optim, args.optimizer)

    if args.optimizer == 'SGD':
        optimizer = optimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        optimizer = optimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = get_scheduler(optimizer, len(train_loader), args)

    cudnn.benchmark = True

    for k, v in sorted(vars(args).items()):
        logger.info(f'{k} = {v}')

    if rank == 0:
        tb_dir = os.path.join(os.path.join(args.output, 'runs'))
        os.makedirs(tb_dir, exist_ok=True)
        train_writer = SummaryWriter(os.path.join(tb_dir, 'train'))
        val_writer = SummaryWriter(os.path.join(tb_dir, 'val'))
    for epoch in range(start_epoch, args.epochs):
        train_losses = {}
        for name in label_names:
            train_losses[name] = AverageMeter()
        train_total_loss = AverageMeter()
        if local_rank != -1:
            train_sampler.set_epoch(epoch)
        model.train()
        for step, (img, label_infos) in enumerate(train_loader):
            img = img.to(device)
            # target = target.long()

            optimizer.zero_grad()
            output = model(img)
            losses_dict, loss = criterion(output, label_infos)
            train_total_loss.update(loss.item(), img.size(0))
            for name in label_names:
                train_losses[name].update(losses_dict[name].item(), img.size(0))
            
            if rank == 0:
                if step % args.print_freq == 0:
                    logger.info(f"Epoch: [{epoch}/{args.epochs}][{step}/{len(train_loader)}], lr: {optimizer.param_groups[0]['lr']:.5f}")
                    logger.info(f"train_total_loss = {train_total_loss.val:.4f}({train_total_loss.avg:.4f})")
                    for name in label_names:
                        train_loss = train_losses[name]
                        logger.info(f"train_loss_{name} = {train_loss.val:.4f}({train_loss.avg:.4f})")

            loss.backward()
            optimizer.step()
            scheduler.step()
        
        train_total_num = torch.tensor(train_total_loss.count).to(img.device)
        train_total_loss_sum = torch.tensor(train_total_loss.sum).to(img.device)
        train_losses_sum = {}
        for name in label_names:
            train_losses_sum[name] = torch.tensor(train_losses[name].sum).to(img.device)

        if rank != -1:
            train_total_num = distributed_sum(train_total_num.clone().detach())
            train_total_loss_sum = distributed_sum(train_total_loss_sum.clone().detach())
            train_loss_avg = train_total_loss_sum / train_total_num
            for name in label_names:
                train_losses_sum[name] = distributed_sum(train_losses_sum[name].clone().detach())

            train_losses_avg = {}
            for name in label_names:
                train_losses_avg[name] = train_losses_sum[name] / train_total_num

        train_loss_avg = train_loss_avg.cpu().numpy()
        train_total_num = train_total_num.cpu().numpy()

        for name in label_names:
            train_losses_avg[name] = train_losses_avg[name].cpu().numpy()

        if rank == 0:
            logger.info(f"[Train all gather loss]")
            logger.info(f"train_total_loss_dist_avg = {train_loss_avg:.4f}, train_total_num = {train_total_num}" )
            for name in label_names:
                train_loss_avg_tmp = train_losses_avg[name]
                logger.info(f"train_loss_dist_avg_{name} = {train_loss_avg_tmp:.4f}")
        if rank == 0:
            # write summary here
            train_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            train_writer.add_scalar('train_total_loss', train_loss_avg, epoch)
            for name in label_names:
                loss_value = train_losses_avg[name]
                train_writer.add_scalar('train_loss_' + name, loss_value, epoch)
        if rank == 0:
            # print accuracy here
            pass
            
        # exit(0)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            val_losses = {}
            for name in label_names:
                val_losses[name] = AverageMeter()
            val_total_loss = AverageMeter()
            model.eval()
            with torch.no_grad():
                eval_pbar = tqdm(val_loader, desc=f'epoch {epoch + 1} / {args.epochs} evaluating', position=1, disable=False if rank in [-1, 0] else True)
                for step, (img, label_infos) in enumerate(eval_pbar):
                    img = img.to(device)
                    output = model(img)
                    losses_dict, loss = criterion(output, label_infos)
                    val_total_loss.update(loss.item(), img.size(0))
                    for name in label_names:
                        val_losses[name].update(losses_dict[name].item(), img.size(0))
                
                val_total_num = torch.tensor(val_total_loss.count).to(img.device)
                val_total_loss_sum = torch.tensor(val_total_loss.sum).to(img.device)
                val_losses_sum = {}
                for name in label_names:
                    val_losses_sum[name] = torch.tensor(val_losses[name].sum).to(img.device)
                
                if rank == 0:
                    logger.info(f"Val Epoch: [{epoch}/{args.epochs}]")
                    logger.info(f"val_total_loss = {val_total_loss.avg:.4f}")
                    
                    for name in label_names:
                        val_loss = val_losses[name]
                        logger.info(f"val_loss_{name} = {val_loss.avg:.4f}")

                if rank != -1:
                    val_total_num = distributed_sum(val_total_num.clone().detach())
                    val_total_loss_sum = distributed_sum(val_total_loss_sum.clone().detach())
                    val_loss_avg = val_total_loss_sum / val_total_num

                    for name in label_names:
                        val_losses_sum[name] = distributed_sum(val_losses_sum[name].clone().detach())

                    val_losses_avg = {}
                    for name in label_names:
                        val_losses_avg[name] = val_losses_sum[name] / val_total_num
    
                val_loss_avg = val_loss_avg.cpu().numpy()
                val_total_num = val_total_num.cpu().numpy()
                for name in label_names:
                    val_losses_avg[name] = val_losses_avg[name].cpu().numpy()
                
                if rank == 0:
                    logger.info(f"[Val all gather loss]")
                    logger.info(f"val_total_loss_dist_avg = {val_loss_avg:.4f}, val_total_num = {val_total_num}" )
                    for name in label_names:
                        val_loss_avg_tmp = val_losses_avg[name]
                        logger.info(f"val_loss_dist_avg_{name} = {val_loss_avg_tmp:.4f}")
     
                    # write summary here
                    val_writer.add_scalar('val_total_loss', val_loss_avg, epoch)
                    for name in label_names:
                        loss_value = val_losses_avg[name]
                        val_writer.add_scalar('val_loss_' + name, loss_value, epoch)

                    save_path = os.path.join(args.output, 'models', f'epoch_{epoch}')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    model_to_save = (model.module if hasattr(model, "module") else model)
                    torch.save({
                        'epoch': epoch,
                        'out_dir': save_path,
                        'state_dict': model_to_save.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        save_path + '.pth')
                    
                    if val_loss_avg.item() < val_min_loss:
                        val_min_loss = val_loss_avg.item()
                        best_epoch = epoch
                        save_path_best = os.path.join(args.output, 'models', 'best.pth')
                        logger.info(f"Save best epoch {best_epoch} with val loss: {val_min_loss}")
                        torch.save({
                            'epoch': best_epoch,
                            'out_dir': save_path_best,
                            'state_dict': model_to_save.state_dict(),
                            'optimizer': optimizer.state_dict()},
                            save_path_best)
                        
                    # torch.save(model_to_save.state_dict(), os.path.join(save_path, f'.pth'))

    if rank == 0:
        train_writer.close()
        val_writer.close()

        
def check_rootfolders(args):
    """Create log and model folder"""
    folders_util = [args.output]
    for folder in folders_util:
        os.makedirs(folder, exist_ok=True)


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
    print(args)
    distributed_init(backend = args.backend)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    args.world_size = int(os.environ["WORLD_SIZE"])

    print(f"[init] == local rank: {local_rank}, global rank: {rank} == devices: {device}")

    train(rank, local_rank, device, args)