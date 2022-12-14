import time
import os
import platform
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from data import BaseCollateFunction
from utils.config import Config
from utils.util import AverageMeter, adjust_learning_rate, format_time, set_seed
from utils.build import build_dataset, build_logger, build_loss, build_optimizer, build_model

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config file path')
    args = parser.parse_args()

    return args
def get_config(args: argparse.Namespace) -> Config:
    cfg = Config.fromfile(args.config)

    cfg.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    cfg.work_dir = os.path.join(cfg.work_dir, f"{cfg.timestamp}")
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # cfgname
    cfg.cfgname = os.path.splitext(os.path.basename(args.config))[0]
    assert cfg.cfgname is not None

    # worker
    cfg.num_workers = min(cfg.num_workers, mp.cpu_count()-2)

    # seed
    if not hasattr(cfg, 'seed'):
        cfg.seed = 42
    set_seed(cfg.seed)

    # resume or load init weights
    # if args.resume:
    #     cfg.resume = args.resume
    # if args.load:
    #     cfg.load = args.load
    # assert not (cfg.resume and cfg.load)

    return cfg

def load_weights(ckpt_path, model, optimizer, resume=True) -> None:
    # load checkpoint
    print("==> Loading checkpoint '{}'".format(ckpt_path))
    assert os.path.isfile(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location='cuda')

    if resume:
        # load model & optimizer
        model.load_state_dict(checkpoint['byol_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    else:
        raise ValueError

    start_epoch = checkpoint['epoch'] + 1
    print("Loaded. (epoch {})".format(checkpoint['epoch']))
    return start_epoch

def main():
    args = get_args()
    cfg = get_config(args)

    world_size = torch.cuda.device_count() 
    print(f'GPUs on this node: {world_size}')
    cfg.world_size = world_size

    log_file = os.path.join(cfg.work_dir, f'{cfg.timestamp}.cfg')

    with open(log_file, 'a') as f:
        f.write(cfg.pretty_text)

    # spawn
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, cfg))

def main_worker(rank, world_size, cfg):
    print(f'==> start rank: {rank}')
    
    local_rank = rank % 8
    cfg.local_rank = local_rank
    torch.cuda.set_device(rank)

    print(f'System : {platform.system()}')
    if platform.system() == 'Windows':
        dist.init_process_group(backend='gloo', init_method=f'tcp://localhost:{cfg.port}',
                            world_size=world_size, rank=rank)
    else: # Linux
        dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{cfg.port}',
                            world_size=world_size, rank=rank)

    # build logger, writer
    logger, writer = None, None
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(cfg.work_dir, 'tensorboard'))
        logger = build_logger(cfg.work_dir, 'pretrain')

    # build data loader
    bsz_gpu = int(cfg.batch_size / cfg.world_size)
    print('batch_size per gpu:', bsz_gpu)

    #build data loader
    train_set = build_dataset(cfg.data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    collate_fn = BaseCollateFunction()
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=bsz_gpu,
        collate_fn = collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    
    # build model, criterion, optimizer
    model = build_model(cfg.backbone, cfg.model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    #??????????????????????????????backward???find_unused_parameters ????????? True
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank], find_unused_parameters=False)

    criterion = build_loss(cfg.loss).cuda()#NegativeCosineSimilarity().cuda()
    optimizer = build_optimizer(cfg.optimizer, model.parameters())

    start_epoch = 1
    if cfg.resume:
        start_epoch = load_weights(cfg.resume, model, optimizer, resume=True)
    cudnn.benchmark = True

    model.train() # ??????batch normalization ??? dropout
    for epoch in range(start_epoch, cfg.epochs + 1):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(cfg.lr_cfg, optimizer, epoch)

        # train; all processes
        train(model, train_loader, criterion, optimizer, epoch, cfg, logger, writer)
        
        # save ckpt; master process
        if rank == 0 and epoch % cfg.save_interval == 0:
            model_path = os.path.join(cfg.work_dir, f'epoch_{epoch}.pth')
            state_dict = {
                'optimizer_state': optimizer.state_dict(),
                'byol_state': model.state_dict(),
                'epoch': epoch
            }
            torch.save(state_dict, model_path)
            
def train(model, dataloader, criterion, optimizer, epoch, cfg, logger=None, writer=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    num_iter = len(dataloader)
    iter_end = time.time()
    epoch_end = time.time()
    for idx, (indice, (x0, x1)) in enumerate(dataloader):
        
        x0 = x0.cuda(non_blocking=True)
        x1 = x1.cuda(non_blocking=True)

        # measure data loading time
        data_time.update(time.time() - iter_end)

        # compute output
        out= model(x0, x1)

        loss = criterion(*out)
        losses.update(loss.item(), x0.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - iter_end)
        iter_end = time.time()

        # print info
        if (idx + 1) % cfg.log_interval == 0 and logger is not None:  # cfg.rank == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch[epoch][idx/iter] [{epoch}][{idx+1}/{num_iter}] - '
                        f'data_time: {data_time.avg:.3f},     '
                        f'batch_time: {batch_time.avg:.3f},     '
                        f'lr: {lr:.5f},     '
                        f'loss(loss avg): {loss:.3f}({losses.avg:.3f})')
        
    if logger is not None: 
        now = time.time()
        epoch_time = format_time(now - epoch_end)
        logger.info(f'Epoch [{epoch}] - epoch_time: {epoch_time}, '
                    f'train_loss: {losses.avg:.3f}')
    
    if writer is not None:
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Pretrain/lr', lr, epoch)
        writer.add_scalar('Pretrain/loss', losses.avg, epoch)
if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()