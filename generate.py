import if_net_texture.models.local_model as model
import if_net_texture.models.dataloader as dataloader
import numpy as np
import argparse
from if_net_texture.models.generation import Generator
import if_net_texture.config.config_loader as cfg_loader
import os
import trimesh
import torch
from if_net_texture.data_processing import utils
from tqdm import tqdm



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generation Model'
    )

    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()

    cfg = cfg_loader.load(args.config)

    net = model.get_models()[cfg['model']]()

    dataloader = dataloader.VoxelizedDataset('test_texture', cfg, generation = True, num_workers=0).get_loader()

    gen = Generator(net, cfg)


    out_path = 'experiments/{}/evaluation_{}/Track_2_test/eval'.format(cfg['folder_name'], gen.checkpoint)


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
        if cfg['generation']['mode'] == 'test_texture' or cfg['generation']['mode'] == 'small_test':
            path_surface = path
        print(scan_name)
        
        #if not os.path.exists(path _surface):
            
        mesh = trimesh.load(path_surface)
        
        #print(type(mesh))
        
        # create new uncolored mesh for color prediction
        pred_mesh = trimesh.Trimesh(mesh.vertices, mesh.faces)
        
        # colors will be attached per vertex
        # subdivide in order to have high enough number of vertices for good texture representation
        #pred_mesh = pred_mesh.subdivide().subdivide()
        
        pred_mesh = pred_mesh.subdivide()

        pred_verts_gird_coords = utils.to_grid_sample_coords( pred_mesh.vertices, cfg['data_bounding_box'])
        pred_verts_gird_coords = torch.tensor(pred_verts_gird_coords).unsqueeze(0)


        colors_pred_surface = gen.generate_colors(inputs, pred_verts_gird_coords)

        # attach predicted colors to the mesh
        pred_mesh.visual.vertex_colors = colors_pred_surface

        #pred_mesh.export( file_out_path + f'{filename_partial}_color_reconstruction.obj')
        pred_mesh.export( file_out_path + f'{scan_name}-completed.obj')
