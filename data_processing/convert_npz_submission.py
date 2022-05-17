import numpy as np
import trimesh

from glob import glob
path = "./experiments/pretrained_texture_regular/evaluation_233/Track_2/eval/model_0/model_0-completed.obj"

#for mesh_path in glob(path + '/*/*.obj'):

    #mesh = trimesh.load(mesh_path)

    #meshes.append(mesh)


mesh = trimesh.load(path)

faces = mesh.faces
vertices = mesh.vertices

num_points = 20000
colored_point_cloud, face_idxs = mesh.sample(num_points, return_index = True)

faces_sampled = mesh.faces[face_idxs]

texcoored_indices = face_idxs

#faces_uvs = mesh.visual.uv[face_vertices]

texture = mesh.visual.material.image


print(texture.shape)

#print(texture)