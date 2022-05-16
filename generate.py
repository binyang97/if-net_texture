import if_net_texture.models.local_model as model
import if_net_texture.models.dataloader as Dataloader
import numpy as np
import argparse
from if_net_texture.models.generation import Generator
import if_net_texture.config.config_loader as cfg_loader
import os
import trimesh
import torch
from if_net_texture.data_processing import utils
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initalize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def generate_basic(rank, world_size, cfg):
    print(f'Running basic DDP on rank {rank}.')
    setup(rank, world_size)

    net = model.get_models()[cfg['model']](rank = rank)
    net = net.to(rank)
    ddp_model = DDP(net, device_ids = [rank])
    if torch.cuda.get_device_name(rank) == "NIVIDIA GeoForce GTX 1080":
        cfg['training']['batch_size'] = int(cfg['training']['batch_size']/11.0*8)

    dataloader = Dataloader.VoxelizedDataset('test_texture', cfg, generation = True, num_workers=0).get_loader()

    gen = Generator(ddp_model, cfg, device=ddp_model.device, rank = rank, world_size = world_size)

    out_path = 'experiments/{}/evaluation_{}/Track_2/eval'.format(cfg['folder_name'], gen.checkpoint)

    # Dataset index shuffle
    data_length = len(dataloader)
    data_partial_length = int(data_length/world_size)
    if not rank:
        data_index = np.arange(0, data_length, dtype=int)
        np.random.shuffle(data_index)
        for i in range(1, world_size):
            partial_index = torch.from_numpy(partial_index[data_partial_length*i:data_partial_length*(i+1)])
            dist.send(tensor=partial_index.to(dtype=torch.int))
        data_index = data_index[: data_partial_length]
    else:
        index_torch = torch.zeros(data_partial_length, dtype=torch.int)
        dist.recv(tensor=index_torch, src=0)
    dist.barrier()

    dataloader.random_split(data_index)

    for data in tqdm(dataloader):

        try:
            inputs = data['inputs']
            path = data['path'][0]
        except:
            print('none')
            continue

        
        #print(path)
        path = os.path.normpath(path)
        challange = path.split(os.sep)[-4]
        split = path.split(os.sep)[-3]
        gt_file_name = path.split(os.sep)[-3]
        scan_name = path.split(os.sep)[-3]
        
        gt_file_name_scaled = path.split(os.sep)[-2]
        basename = path.split(os.sep)[-1]
        filename_partial = os.path.splitext(path.split(os.sep)[-1])[0]

        file_out_path = out_path + '/{}/'.format(scan_name)
        os.makedirs(file_out_path, exist_ok=True)

        if os.path.exists(file_out_path + 'colored_surface_reconstuction.obj'):
            continue


        path_surface = os.path.join(cfg['data_path'], split, gt_file_name, gt_file_name + '_normalized.obj')
        if cfg['generation']['mode'] == 'test_texture':
            path_surface = path
        print(scan_name)
        mesh = trimesh.load(path_surface, force = 'mesh')
        
        # create new uncolored mesh for color prediction
        pred_mesh = trimesh.Trimesh(mesh.vertices, mesh.faces)
        
        # colors will be attached per vertex
        # subdivide in order to have high enough number of vertices for good texture representation
        pred_mesh = pred_mesh.subdivide().subdivide()

        pred_verts_gird_coords = utils.to_grid_sample_coords( pred_mesh.vertices, cfg['data_bounding_box'])
        pred_verts_gird_coords = torch.tensor(pred_verts_gird_coords).unsqueeze(0)


        colors_pred_surface = gen.generate_colors(inputs, pred_verts_gird_coords)

        # attach predicted colors to the mesh
        pred_mesh.visual.vertex_colors = colors_pred_surface

        pred_mesh.export( file_out_path + f'{scan_name}-completed.obj')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generation Model'
    )
    #parser.add_argument('config', type=int, default=1, help='number of GPUs')
    parser.add_argument('config', type=str, help='Path to config file.')
    
    args = parser.parse_args()

    cfg = cfg_loader.load(args.config)
    
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    
    world_size = n_gpus
    
    mp.spawn(generate_basic,
             args=(world_size, cfg),
             nprocs=world_size,
             join=True)
             
   # generate_basic(rank, world_size, cfg)

    # net = model.get_models()[cfg['model']]()

    # dataloader = dataloader.VoxelizedDataset('test', cfg, generation = True, num_workers=0).get_loader()

    # gen = Generator(net, cfg)


    # out_path = 'experiments/{}/evaluation_{}/'.format(cfg['folder_name'], gen.checkpoint)


    # for data in tqdm(dataloader):


    #     try:
    #         inputs = data['inputs']
    #         path = data['path'][0]
    #     except:
    #         print('none')
    #         continue


    #     path = os.path.normpath(path)
    #     challange = path.split(os.sep)[-4]
    #     split = path.split(os.sep)[-3]
    #     gt_file_name = path.split(os.sep)[-2]
    #     basename = path.split(os.sep)[-1]
    #     filename_partial = os.path.splitext(path.split(os.sep)[-1])[0]

    #     file_out_path = out_path + '/{}/'.format(gt_file_name)
    #     os.makedirs(file_out_path, exist_ok=True)

    #     if os.path.exists(file_out_path + 'colored_surface_reconstuction.obj'):
    #         continue


    #     path_surface = os.path.join(cfg['data_path'], split, gt_file_name, gt_file_name + '_normalized.obj')

    #     mesh = trimesh.load(path_surface)
        
    #     # create new uncolored mesh for color prediction
    #     pred_mesh = trimesh.Trimesh(mesh.vertices, mesh.faces)
        
    #     # colors will be attached per vertex
    #     # subdivide in order to have high enough number of vertices for good texture representation
    #     pred_mesh = pred_mesh.subdivide().subdivide()

    #     pred_verts_gird_coords = utils.to_grid_sample_coords( pred_mesh.vertices, cfg['data_bounding_box'])
    #     pred_verts_gird_coords = torch.tensor(pred_verts_gird_coords).unsqueeze(0)


    #     colors_pred_surface = gen.generate_colors(inputs, pred_verts_gird_coords)

    #     # attach predicted colors to the mesh
    #     pred_mesh.visual.vertex_colors = colors_pred_surface

