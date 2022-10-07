
"""
copy from https://github.com/xyupeng/ContrastiveCrop/blob/6bd54c433dd05e817cd3d3adcd7d61b4bb12c811/builder/build.py#L30

Build optimizers and schedulers
"""
import torch
import os
import logging
import torch
import torch.nn as nn
import torchvision
import torchvision.models
import torch.nn as nn
import torchvision.transforms as T

from data.transforms import transform
from models.downstream import DownStream
from utils.config import ConfigDict
import models
from models.byol import BYOL
from models.nnclr import NNCLR
from models.simsiam import SimSiam
from models.simclr import SimCLR
from models.barlow_twins import BarlowTwins
from models.our import OUR
import losses
import data

def build_transform(cfg: ConfigDict):
    args = cfg.copy()
    name = args.pop('type')
    return data.transforms.__dict__[name](**args)

def build_dataset(cfg: ConfigDict):
    args = cfg.copy()
    name = args.pop('type')
    transform_args = args.pop('transform')
    
    args['transform'] = build_transform(transform_args)
    if hasattr(torchvision.datasets, name):
        dataset = getattr(torchvision.datasets, name)(**args)
    else:
        dataset = data.__dict__[name](**args)
    return dataset

def build_model(backbone_cfg: ConfigDict, model_cfg: ConfigDict) -> nn.Module:
    """
    Examples:
        >>> module = build_model(cfg1, cfg2)
    """
    backbone_args = backbone_cfg.copy()
    backbone_name = backbone_args.pop('type')

    model_args = model_cfg.copy()
    model_name = model_args.pop('type')
    model = None
    if model_name == 'BYOL':
        resnet_q = models.__dict__[backbone_name](**backbone_args)
        resnet_k = models.__dict__[backbone_name](**backbone_args)
        backbone_q = nn.Sequential(*list(resnet_q.children())[:-1])
        backbone_k = nn.Sequential(*list(resnet_k.children())[:-1])
        model = BYOL(backbone_q, backbone_k)

    elif model_name == 'NNCLR':
        resnet_q = models.__dict__[backbone_name](**backbone_args)
        backbone_q = nn.Sequential(*list(resnet_q.children())[:-1])
        model = NNCLR(backbone_q)
    elif model_name == 'SimCLR':
        resnet_q = models.__dict__[backbone_name](**backbone_args)
        backbone_q = nn.Sequential(*list(resnet_q.children())[:-1])
        model = SimCLR(backbone_q)
    elif model_name == 'SimSiam':
        resnet_q = models.__dict__[backbone_name](**backbone_args)
        backbone_q = nn.Sequential(*list(resnet_q.children())[:-1])
        model = SimSiam(backbone_q)

    elif model_name == 'BarlowTwins':
        resnet_q = models.__dict__[backbone_name](**backbone_args)
        backbone_q = nn.Sequential(*list(resnet_q.children())[:-1])
        model = BarlowTwins(backbone_q)

    elif model_name == 'OUR':
        resnet_q = models.__dict__[backbone_name](**backbone_args)
        backbone_q = nn.Sequential(*list(resnet_q.children())[:-1])
        model = OUR(backbone_q)

    elif model_name == 'Linear':
        backbone = models.__dict__[backbone_name](**backbone_args)
        backbone = nn.Sequential(*list(backbone.children())[:-1])
        model = DownStream(backbone, model_args)
    return model


def build_optimizer(cfg, params):
    # cfg: ConfigDict
    # TODO: print(type(cfg))
    args = cfg.copy()
    name = args.pop('type')
    if hasattr(torch.optim, name):
        optimizer = getattr(torch.optim, name)(params, **args)
    else:
        raise ValueError(f'torch.optim has no optimizer \'{name}\'.')
    return optimizer

def build_loss(cfg: ConfigDict):
    args = cfg.copy()
    name = args.pop('type')
    criterion = losses.__dict__[name](**args)
    return criterion
    
def build_scheduler(cfg, **kwargs):
    # cfg: ConfigDict
    args = cfg.copy()
    name = args.pop('type')
    if hasattr(torch.optim.lr_scheduler, name):
        scheduler = getattr(torch.optim.lr_scheduler, name)(**kwargs, **args)
    else:
        raise ValueError(f'torch.optim.lr_scheduler has no scheduler\'{name}\'.')
    return scheduler


def build_logger(work_dir, cfgname):
    log_file = cfgname + '.log'
    log_path = os.path.join(work_dir, log_file)

    logger = logging.getLogger(cfgname)
    logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler1 = logging.FileHandler(log_path)
    handler1.setFormatter(formatter)

    handler2 = logging.StreamHandler()
    handler2.setFormatter(formatter)

    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.propagate = False

    return logger