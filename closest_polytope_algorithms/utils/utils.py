import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import Voronoi
from pypolycontain.utils.random_polytope_generator import get_k_random_edge_points_in_zonotope_OverR3T
# from pypolycontain.utils.random_polytope_generator import get_k_random_edge_points_in_zonotope
from pypolycontain.lib.operations import to_AH_polytope
from r3t.common.help import *


def build_key_point_kd_tree_OverR3T(polytopes, generator_idx_list, k_lists, key_vertex_count = 0, distance_scaling_array=None):

    assert polytopes[0].__name__=='zonotope', "Not a Zonotope!"
    p0_projected = project_zonotope(polytopes[0], dim=[0, 1], mode='full')
    dim = p0_projected.x.shape[0]
    key_point_to_zonotope_map = dict()
    scaled_key_points = np.zeros((len(polytopes)*key_vertex_count,dim))
    if distance_scaling_array is None:
        distance_scaling_array = np.ones(n)
        
    p_idx = 0
    for i, k_list in enumerate(k_lists):
        for j, k in enumerate(k_list):
            p = polytopes[p_idx]
            # p_projected = project_zonotope(p, dim=[0, 1], mode='full')
            # scaled_key_points[p_idx*(1+key_vertex_count),:] = np.multiply(distance_scaling_array, p_projected.x[:, 0], dtype='float')
            # key_point_to_zonotope_map[p_projected.x[:, 0].tostring()]=[[p], k]
            # slice by k
            # key_vertex_count: number of keypoints for zonotope
            other_key_points, keypoint_k_lists = get_k_random_edge_points_in_zonotope_OverR3T(p, generator_idx_list[i][j], N=key_vertex_count, k=k, lk=-0.5, uk=0.5) 
            scaled_other_key_points = np.multiply(other_key_points, distance_scaling_array, dtype='float')
            scaled_key_points[p_idx*key_vertex_count:(p_idx+1)*key_vertex_count, :] = scaled_other_key_points

            for j, kp in enumerate(other_key_points):
                key_point_to_zonotope_map[kp.tostring()] = [[p], keypoint_k_lists[j]]

            p_idx += 1

    return KDTree(scaled_key_points), key_point_to_zonotope_map



def build_polyotpe_centroid_voronoi_diagram(polytopes):
    n = len(polytopes)
    if polytopes[0].type=='AH_polytope':
        k = polytopes[0].t.shape[0]
    elif polytopes[0].type=='zonotope':
        k = polytopes[0].x.shape[0]
    else:
        raise NotImplementedError
    centroids = np.zeros((n, k))
    for i, z in enumerate(polytopes):
        if polytopes[0].type == 'AH_polytope':
            centroids[i, :] = polytopes[i].t[:,0]
        elif polytopes[0].type == 'zonotope':
            centroids[i, :] = polytopes[i].x[:,0]
        else:
            raise NotImplementedError
    return Voronoi(centroids)
