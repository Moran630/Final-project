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

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def deploy(args, model_path):
    logger = init_logger(log_file=args.output + f'/log_deploy', rank=-1)
    # assert args.model_name == 'SWINUNETR'
    if args.model_name == 'resnet18':
        model = ResNet18(num_classes=args.num_classes, num_channels=args.num_channels)
    elif args.model_name == 'resnet18_rnn':
        model = ResNet18_rnn(num_classes=args.num_classes, num_channels=args.num_channels,  mode='deploy')
    else:
        print(args.model, 'not support!')
        raise
    print(model)
    
    
    
    if args.init_weights == '':
        logger.warning(f"the checkpoint if null!")
        raise
    else:
        device = torch.device("cuda", 0)
        logger.info(f"Loading weight from {args.init_weights}")
        checkpoint = torch.load(args.init_weights, map_location=device)
        epoch = checkpoint['epoch']
        state = model.state_dict()
        state.update(checkpoint['state_dict'])
        model.load_state_dict(state, strict=True)  # , strict=False
    
    model.to(device)
    model.eval()
    set_requires_grad(model)

    test_dataset = DefaultLoaderCls5Rnn_test(args.test_db, args, logger=logger, mode='test', input_size=args.input_size)
    demo_data = test_dataset[0]
    img_tensor, label, uid = demo_data
    img_tensor = img_tensor.unsqueeze(0).to(device)
    print(img_tensor.size())
    with torch.no_grad():
        print(model(img_tensor))
        traced_model = torch.jit.trace(model, (img_tensor, ))
    # print(traced_model.graph)
    traced_model.save(model_path)

    model_deploy = torch.jit.load(model_path)
    with torch.no_grad():
        probs = model_deploy(img_tensor)
        print(uid)
        print(probs)


if __name__  == '__main__':
    args = parser.parse_args()
    print(args)
    model_path = args.output_pt
    deploy(args, model_path)
