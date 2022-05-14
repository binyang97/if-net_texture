from glob import glob
import os
import tqdm
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import trimesh

import if_net_texture.config.config_loader as cfg_loader



print('Finding Paths to convert (from .npz to .obj files).')
paths = glob('../track2_testdata/*/*/*.npz')


print('Start converting.')
#def convert(path):
    #outpath = path[:-4] + '.obj'

    #cmd = 'python -m sharp convert {} {}'.format(path,outpath)
    #os.system(cmd)

def scale_back(parameters):
    partial_mesh_fullpath, generation_mesh_fullpath= parameters

    mesh_generation = trimesh.load(generation_mesh_fullpath)
    mesh_partial = trimesh.load(partial_mesh_fullpath)

    output_fullpath = os.path.splitext(generation_mesh_fullpath)[0] + "_scaled_back" + ".obj"

    total_size = (mesh_partial.bounds[1] - mesh_partial.bounds[0]).max()
    centers = (mesh_generation.bounds[1] + mesh_generation.bounds[0]) /2

    mesh_generation.apply_translation(-centers)
    mesh_generation.apply_scale(total_size)
    
    mesh_generation.export(output_fullpath)
    print(f'Finished to scale back the output{output_fullpath}')


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Run color sampling. Samples surface points on the GT objects, and saves their coordinates along with the RGB color at their location.'
    )

    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()

    cfg = cfg_loader.load(args.config)
    if cfg['processing']['mode'] == 'test':

        generation_mesh_paths = glob(cfg['processing']['scale_back_obj']['generation_path'] + cfg['processing']['scale_back_obj']['input_file_regix'])
        partial_mesh_paths = glob(cfg['data_path'] + cfg['processing']['voxelized_colored_pointcloud_sampling']['input_file_regix'])

        params = []

        for i, generation_path in enumerate(generation_mesh_paths):
            params.append([generation_path, partial_mesh_paths[i]])




    p = Pool(mp.cpu_count())
    for _ in tqdm.tqdm(p.imap_unordered(scale_back, params), total=len(params)):
        pass
