from if_net_texture.data_processing import utils
from scipy.spatial import cKDTree as KDTree
import numpy as np
import trimesh
from glob import glob
import os
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import random

import sys
from if_net_texture.config import config_loader as cfg_loader
import traceback
import tqdm
import if_net_texture.data_processing.utils


def voxelized_colored_pointcloud_sampling(tmp_path):
    partial_mesh_path, grid_points, kdtree, bbox, res, num_points, bbox_str, generation_mesh_fullpath, test= tmp_path
    try:
        path = os.path.normpath(partial_mesh_path)
        gt_file_name = path.split(os.sep)[-2] # model 2
        full_file_name = path.split(os.sep)[-1][:-4] # model_2-partial_01

        out_file = os.path.dirname(partial_mesh_path) + '/{}_voxelized_colored_point_cloud_res{}_points{}_bbox{}.npz'\
            .format(full_file_name, res, num_points, bbox_str)
        
        print("gt_file_name:", gt_file_name)
        print("partial_path: ", os.path.dirname(partial_mesh_path))

        
        #if os.path.exists(out_file):
            #print('File exists. Done.')
            #return
        
        # color from partial input
        partial_mesh = utils.as_mesh(trimesh.load(partial_mesh_path))
        colored_point_cloud, face_idxs = partial_mesh.sample(num_points, return_index = True)

        triangles = partial_mesh.triangles[face_idxs]
        face_vertices = partial_mesh.faces[face_idxs]
        faces_uvs = partial_mesh.visual.uv[face_vertices]

        q = triangles[:, 0]
        u = triangles[:, 1]
        v = triangles[:, 2]

        uvs = []

        for i, p in enumerate(colored_point_cloud):
            barycentric_weights = utils.barycentric_coordinates(p, q[i], u[i], v[i])
            uv = np.average(faces_uvs[i], 0, barycentric_weights)
            uvs.append(uv)

        partial_texture = partial_mesh.visual.material.image

        colors = trimesh.visual.color.uv_to_color(np.array(uvs), partial_texture)

        R = - 1 * np.ones(len(grid_points), dtype=np.int16)
        G = - 1 * np.ones(len(grid_points), dtype=np.int16)
        B = - 1 * np.ones(len(grid_points), dtype=np.int16)

        _, idx = kdtree.query(colored_point_cloud)
        R[idx] = colors[:,0]
        G[idx] = colors[:,1]
        B[idx] = colors[:,2]

        # encode uncolorized, complete shape of object (at inference time obtained from IF-Nets surface reconstruction)
        # encoding is done by sampling a pointcloud and voxelizing it (into discrete grid for 3D CNN usage)
        
        if test:
            full_shape = trimesh.load(generation_mesh_fullpath)
        else:
            dir = os.path.normpath(os.path.dirname(partial_mesh_path))
            dir_comp = dir.split(os.sep)
            dir_comp[-2] = dir_comp[-2][:-7] + 'gt'
            full_shape = utils.as_mesh(trimesh.load(os.path.join(os.sep.join(dir_comp), gt_file_name +'.obj')))
            
        shape_point_cloud = full_shape.sample(num_points)
        S = np.zeros(len(grid_points), dtype=np.int8)

        _, idx = kdtree.query(shape_point_cloud)
        S[idx] = 1

        np.savez(out_file, R=R, G=G,B=B, S=S,  colored_point_cloud=colored_point_cloud, bbox = bbox, res = res)

    except Exception as err:
        print('Error with {}: {}'.format(partial_mesh_path, traceback.format_exc()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generates the input for the network: a partial colored shape and a uncolorized, but completed shape. \
        Both encoded as 3D voxel grids for usage with a 3D CNN.'
    )

    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()

    cfg = cfg_loader.load(args.config)

    # shorthands
    bbox = cfg['data_bounding_box']
    res = cfg['input_resolution']
    num_points = cfg['input_points_number']
    bbox_str = cfg['data_bounding_box_str']

    grid_points = utils.create_grid_points_from_xyz_bounds(*bbox, res)
    kdtree = KDTree(grid_points)

    print('Fining all input partial paths for voxelization.')
    paths = sorted(glob(cfg['data_path'] + cfg['preprocessing']['voxelized_colored_pointcloud_sampling']['input_files_regex']))
    
    test = cfg['preprocessing']['voxelized_colored_pointcloud_sampling']['evaluation']
    if test:
        generation_mesh_paths = sorted(glob(cfg['preprocessing']['scale_back_obj']['generation_path'] + cfg['preprocessing']['scale_back_obj']['input_files_regex']))
        print(len(generation_mesh_paths), len(paths))
    #print(paths)
        new_paths = []
        for i, path in enumerate(paths):
            new_paths.append((path, grid_points, kdtree, bbox, res, num_points, bbox_str, generation_mesh_paths[i], test))
            
    else:
        new_paths = []
        for i, path in enumerate(paths):
            new_paths.append((path, grid_points, kdtree, bbox, res, num_points, bbox_str, None, test))
    
    #print(new_paths)
    print('Start voxelization.')
    p = Pool(mp.cpu_count())
    for _ in tqdm.tqdm(p.imap_unordered(voxelized_colored_pointcloud_sampling, new_paths), total=len(paths)):
        pass
    p.close()
    p.join()
    
