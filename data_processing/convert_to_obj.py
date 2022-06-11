from glob import glob
import os
import tqdm
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import trimesh
import time

import if_net_texture.config.config_loader as cfg_loader



#print('Finding Paths to convert (from .npz to .obj files).')
#paths = glob('../track2_testdata/*/*/*.npz')


#print('Start converting.')
#def convert(path):
    #outpath = path[:-4] + '.obj'

    #cmd = 'python -m sharp convert {} {}'.format(path,outpath)
    #os.system(cmd)

def scale_back(parameters):
    partial_mesh_fullpath, generation_mesh_fullpath= parameters

    mesh_generation = trimesh.load(generation_mesh_fullpath)
    
    mesh_partial = trimesh.load(partial_mesh_fullpath)
    
    output_fullpath = os.path.splitext(generation_mesh_fullpath)[0] + "_normal" + ".obj"
    if os.path.exists(output_fullpath):
        mesh = trimesh.load(output_fullpath)
        #print(type(mesh))
        #print(output_fullpath)
        if isinstance(mesh, trimesh.Trimesh):
            print('File exists. Done.')
            return
        else:
            print("File exists but is not Mesh, it will retry")
    
    OUTPUT = False
    while OUTPUT:
        print("resized")
        total_size = (mesh_partial.bounds[1] - mesh_partial.bounds[0]).max()
        centers = (mesh_partial.bounds[1] + mesh_partial.bounds[0]) /2
  
        mesh_generation.apply_scale(total_size)
        mesh_generation.apply_translation(centers)
        
        #print(type(mesh_generation))
        if isinstance(mesh_generation, trimesh.Trimesh):
            print("The output is not Mesh, it will retry")
            OUTPUT = False
    
    mesh_generation.export(output_fullpath)
    print(f'Finished to scale back the output{output_fullpath}')


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Run color sampling. Samples surface points on the GT objects, and saves their coordinates along with the RGB color at their location.'
    )

    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()

    cfg = cfg_loader.load(args.config)
    if cfg['preprocessing']['scale_back_obj']['mode'] == 'test':

        generation_mesh_paths = sorted(glob(cfg['preprocessing']['scale_back_obj']['generation_path'] + cfg['preprocessing']['scale_back_obj']['input_files_regex']))
        partial_mesh_paths = sorted(glob(cfg['data_path'] + cfg['preprocessing']['voxelized_colored_pointcloud_sampling']['input_files_regex']))
        

        params = []

        for i, generation_path in enumerate(generation_mesh_paths):
            params.append([partial_mesh_paths[i], generation_path])

    else:
        raise 'The data preprocessing step is only for test set'

    #print(params)
    #for param in params:
        #print(param)
        #time.sleep(2)

    p = Pool(mp.cpu_count())
    for _ in tqdm.tqdm(p.imap_unordered(scale_back, params), total=len(params)):
        pass
