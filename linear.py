import os
import argparse
import platform
import time

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from models.downstream import DownStream

from utils.build import build_dataset, build_logger, build_model, build_optimizer
from utils.config import Config
from utils.util import AverageMeter, TrackMeter, accuracy, format_time, adjust_learning_rate


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config file path')
    args = parser.parse_args()

    return args

def get_config(args: argparse.Namespace) -> Config:
    cfg = Config.fromfile(args.config)

    # work_dir
    os.makedirs(cfg.work_dir, exist_ok=True)

    # cfgname
    cfg.cfgname = os.path.splitext(os.path.basename(args.config))[0]
    assert cfg.cfgname is not None
    # # resume or load init weights
    # if args.resume:
    #     cfg.resume = args.resume
    # if args.load:
    #     cfg.load = args.load
    # assert not (cfg.resume and cfg.load)
    return cfg

def load_weights(
    ckpt_path: str, 
    model: nn.Module, 
    optimizer:nn.Module, 
    resume:bool = False):

    # load checkpoint 
    print(f"==> Loading Checkpoint {ckpt_path}")
    assert os.path.isfile(ckpt_path), 'file is not exist'
    ckpt = torch.load(ckpt_path, map_location='cuda')

    if resume:
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
    else:
        if 'byol_state' in ckpt.keys():
            state_dict = ckpt['byol_state']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.backbone.'):
                    new_state_dict[k] = v
        del state_dict

        msg = model.load_state_dict(new_state_dict, strict=False)
        assert set(msg.missing_keys) == {'module.fc.weight', 'module.fc.bias'}, set(msg.missing_keys)

    start_epoch = ckpt['epoch'] + 1
    print(f"Model weights loaded. (epoch{ckpt['epoch']})")

    return start_epoch

def train(model, dataloader, criterion, optimizer, epoch, cfg, logger, writer):
    """one epoch training"""
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    num_iter = len(dataloader)
    iter_end = time.time()
    epoch_end = time.time()

    for idx, (images, targets) in enumerate(dataloader):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        batch_size = targets.shape[0]

        # compute loss
        logits = model(images)
        loss = criterion(logits, targets)
        acc1, acc5 = accuracy(logits, targets, topk = (1, 5))

        # update metric
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - iter_end)
        iter_end = time.time()

        # print info
        if (idx + 1) % cfg.log_interval == 0 and logger is not None:
                lr = optimizer.param_groups[0]['lr']
                logger.info(f'Epoch [{epoch}][{idx+1}/{num_iter}] - '
                            f'batch_time: {batch_time.avg:.3f},     '
                            f'lr: {lr:.5f},     '
                            f'loss: {losses.avg:.3f},     '
                            f'Acc@1: {top1.avg:.3f}')
    
    epoch_time = format_time(time.time() - epoch_end)
    if logger is not None:
        logger.info(f'Epoch [{epoch}] - epoch_time:{epoch_time}, '
                    f'train_loss: {losses.avg:.3f}, '
                    f'train_Acc@1: {top1.avg:.3f}')

    if writer is not None:
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Linear/lr', lr, epoch)
        writer.add_scalar('Linear/train/loss', losses.avg, epoch)
        writer.add_scalar('Linear/train/acc', top1.avg, epoch)
    
    return losses.avg, top1.avg

def test(model, dataloader, criterion, epoch, logger, writer):
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()

    with torch.no_grad():
        for idx, (images, targets) in enumerate(dataloader):
            images = images.float().cuda()
            targets = targets.cuda()
            batch_size = targets.shape[0]

            # forward
            logits = model(images)
            loss = criterion(logits, targets)
            acc1, acc5 = accuracy(logits, targets, topk=(1,5))

            # update metric
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)

    epoch_time = format_time(time.time() - end)

    if logger is not None:
        logger.info(f'Epoch [{epoch}] - epoch_time: {epoch_time}, '
                    f'test_loss: {losses.avg:.3f}, '
                    f'test_Acc@1: {top1.avg: .3f}')
    if writer is not None:
        writer.add_scalar('Linear/test/loss', losses.avg, epoch)
        writer.add_scalar('Linear/test/acc', top1.avg, epoch)
    
    return losses.avg, top1.avg

def main():
    args = get_args()
    cfg = get_config(args)
    world_size = torch.cuda.device_count()
    print(f'GPUs on this: {world_size}')
    cfg.world_size = world_size

    mp.spawn(main_worker, nprocs=world_size, args=(world_size, cfg))

def main_worker(rank:int, world_size:int, cfg:Config):
    print(f'==> Start rank:{rank}')

    local_rank = rank % world_size
    cfg.local_rank = local_rank
    torch.cuda.set_device(local_rank)

    if platform.system() == 'Windows':
        dist.init_process_group(backend='gloo', init_method=f'tcp://localhost:{cfg.port}',
                            world_size=world_size, rank=rank)
    else: # Linux
        dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{cfg.port}',
                            world_size=world_size, rank=rank)

    # logger
    logger = None
    writer = None
    if rank == 0:
        logger = build_logger(cfg.work_dir, 'linear')
        writer = SummaryWriter(log_dir = os.path.join(cfg.work_dir, 'tensorboard'))
    
    # build data loader
    bsz_gpu = int(cfg.batch_size / cfg.world_size)
    print(f'batch_size per gpu: {bsz_gpu}')

    train_set = build_dataset(cfg.data.train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle = True)
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=bsz_gpu,
        num_workers = cfg.num_workers,
        pin_memory = True,
        sampler=train_sampler,
        drop_last=True
    )

    test_set = build_dataset(cfg.data.test)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size = bsz_gpu,
        num_workers = cfg.num_workers,
        pin_memory=True,
        drop_last = True
    )

    # buiel model and criterion
    model = build_model(cfg.backbone, cfg.model)
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    model.cuda()
    assert len(parameters) == 2  # fc.weight, fc.bias
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank])
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = build_optimizer(cfg.optimizer, parameters)

    start_epoch = 1
    if cfg.resume:
        start_epoch = load_weights(cfg.resume, model, optimizer, resume=True)
    elif cfg.load:
        load_weights(cfg.load, model, optimizer, resume=False)
    cudnn.benchmark = True
    
    
    print(f"==> Start tringing ....")
    model.eval() # 不開啟batch normalization 和 dropout
    test_meter = TrackMeter()
    for epoch in range(start_epoch, cfg.epochs + 1):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(cfg.lr_cfg, optimizer, epoch)

        #train
        train(model, train_loader, criterion, optimizer, epoch, cfg, logger, writer)

        # test
        test_loss, test_acc = test(model, test_loader, criterion, epoch, logger, writer)
        if test_acc > test_meter.max_val and rank == 0:
            model_path = os.path.join(cfg.work_dir, f'best_{cfg.cfgname}.pth')
            state_dict={
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'acc': test_acc,
                'epoch':epoch
            }
            torch.save(state_dict, model_path)

    test_meter.update(test_acc, idx=epoch)
    if rank == 0:
        logger.info(f'Best acc: {test_meter.max_val:.2f} (epoch={test_meter.max_idx})')
if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn')
    main()