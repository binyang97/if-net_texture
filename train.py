import if_net_texture.models.local_model as model
import if_net_texture.models.dataloader as dataloader
from if_net_texture.models import training
import argparse
import torch
import if_net_texture.config.config_loader as cfg_loader
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import random
import time

def setup(rank, world_size):
    os.enviro['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_basic(rank, world_size, cfg):
    # rank: the current process
    # world_size: the number of processors
    print(f'Running basic DDP on rank {rank}.')
    setup(rank, world_size)

    net = model.get_models()[cfg['model']](rank = rank)
    net = net.to(rank)
    ddp_model = DDP(net, device_ids = [rank])
    if torch.cuda.get_device_name(rank) == "NIVIDIA GeoForce GTX 1080":
        cfg['training']['batch_size'] = int(cfg['training']['batch_size']/11.0*8)

    train_dataset = dataloader.VoxelizedDataset('train', cfg, world_size = world_size, rank = rank)
    val_dataset = dataloader.VoxelizedDataset('val', cfg, world_size = world_size, rank = rank)

    # Train index shuffle
    train_length = len(train_dataset)
    train_partial_length = int(train_length/world_size)
    if not rank:
        # if rank is 0
        train_index = np.arange(0, train_length, dtype=int)
        np.random.shuffle(train_index)
        for i in range(1, world_size):
            partial_index = torch.from_numpy(train_index[train_partial_length*i: train_partial_length*(i+1)])
            dist.send(tensor=partial_index.to(dtype=torch.int))
        train_index = train_index[: train_partial_length]
    else:
        index_torch = torch.zeros(train_partial_length, dtype=torch.int)
        dist.recv(tensor=index_torch, src=0)
        train_index = index_torch.detach().cpu().numpy()
    dist.barrier()

    # Validation index shuffle
    val_length = len(val_dataset)
    val_partial_length = int(val_length/world_size)
    if not rank:
        val_index = np.arange(0,val_length, dtype = int)
        np.random.shuffle(val_index)
        for i in range(1, world_size):
            partial_index = torch.from_numpy(val_index[val_partial_length*i: val_partial_length*(i+1)])
            dist.send(tensor=partial_index.to(dtype = torch.int), dst=i)
        val_index = val_index[: val_partial_length]
    else:
        index_torch = torch.zeros(val_partial_length,dtype=torch.int)
        dist.recv(tensor=index_torch, src=0)
        val_index = index_torch.detach().cpu().numpy()
    dist.barrier()

    train_dataset.random_split(train_index)
    val_dataset.random_split(val_index)

    trainer = training.Trainer(ddp_model, ddp_model.device,train_dataset, val_dataset, cfg['folder_name'], optimizer=cfg['training']['optimizer'])
    dist.barrier()
    trainer.train_model(1500)

    cleanup()

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    parser = argparse.ArgumentParser(
        description='Run Model'
    )

    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()


    cfg = cfg_loader.load(args.config)

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus

    processes = []

    random.seed(time.time())

    mp.spwan(train_basic,
            args=(world_size, cfg),
            nprocs=world_size,
            join=True)
    print("main finished")

    # net = model.get_models()[cfg['model']]()

    # train_dataset = dataloader.VoxelizedDataset('train', cfg)

    # val_dataset = dataloader.VoxelizedDataset('val', cfg)

    # trainer = training.Trainer(net,torch.device("cuda"),train_dataset, val_dataset, cfg['folder_name'], optimizer=cfg['training']['optimizer'])
    # trainer.train_model(1500)
