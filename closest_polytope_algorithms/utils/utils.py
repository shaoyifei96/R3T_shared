import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import Voronoi
from pypolycontain.utils.random_polytope_generator import get_k_random_edge_points_in_zonotope_OverR3T
# from pypolycontain.utils.random_polytope_generator import get_k_random_edge_points_in_zonotope
from pypolycontain.lib.operations import to_AH_polytope
from r3t.common.help import *

def build_key_point_kd_tree(polytopes, key_vertex_count = 0, distance_scaling_array=None):
    if key_vertex_count > 0:
        n = len(polytopes)*(1+2**key_vertex_count)
    else:
        n = len(polytopes)
    if polytopes[0].__name__=='AH_polytope':
        dim = polytopes[0].t.shape[0]
    elif polytopes[0].__name__=='zonotope':
        dim = polytopes[0].x.shape[0]
    else:
        raise NotImplementedError
    key_point_to_zonotope_map = dict()
    scaled_key_points = np.zeros((n,dim))
    if distance_scaling_array is None:
        distance_scaling_array = np.ones(n)
    for i, p in enumerate(polytopes):
        if p.__name__=='AH_polytope' and key_vertex_count==0:
            scaled_key_points[i,:] = np.multiply(distance_scaling_array, p.t[:, 0], dtype='float')
            key_point_to_zonotope_map[str(p.t[:, 0])]=[p]
        elif p.__name__ == 'zonotope' and key_vertex_count==0:
            scaled_key_points[i,:] = np.multiply(distance_scaling_array, p.x[:, 0], dtype='float')
            key_point_to_zonotope_map[str(p.x[:, 0])]=[p]
        elif p.__name__=='zonotope':
            scaled_key_points[i*(1+2**key_vertex_count),:] = np.multiply(distance_scaling_array, p.x[:, 0], dtype='float')
            key_point_to_zonotope_map[str(p.x[:, 0])]=[p]
            other_key_points = get_k_random_edge_points_in_zonotope(p, key_vertex_count)
            scaled_other_key_points = np.multiply(distance_scaling_array, other_key_points, dtype='float')
            scaled_key_points[i * (2 ** key_vertex_count + 1) + 1:(i + 1) * (2 ** key_vertex_count + 1),
            :] = scaled_other_key_points
            for kp in other_key_points:
                key_point_to_zonotope_map[str(kp)] = [p]
        else:
            raise NotImplementedError
    return KDTree(scaled_key_points), key_point_to_zonotope_map


def build_key_point_kd_tree_OverR3T(polytopes, generator_idx_list, key_vertex_count = 0, distance_scaling_array=None):

    assert polytopes[0].__name__=='zonotope', "Not a Zonotope!"
    p0_projected = project_zonotope(polytopes[0], dim=[0, 1], mode='full')
    dim = p0_projected.x.shape[0]

    if key_vertex_count > 0:
        n = len(polytopes)*(1+2**key_vertex_count)
    else:
        n = len(polytopes)

    key_point_to_zonotope_map = dict()
    scaled_key_points = np.zeros((n,dim))
    if distance_scaling_array is None:
        distance_scaling_array = np.ones(n)
        
    for i, p in enumerate(polytopes):
        assert p.__name__=='zonotope', "Not Zonotope in kd Tree!"
        if p.__name__=='AH_polytope' and key_vertex_count==0:
            scaled_key_points[i,:] = np.multiply(distance_scaling_array, p.t[:, 0], dtype='float')
            key_point_to_zonotope_map[str(p.t[:, 0])]=[p]
        elif p.__name__ == 'zonotope' and key_vertex_count==0:
            p_projected = project_zonotope(p, dim=[0, 1], mode='full')
            scaled_key_points[i,:] = np.multiply(distance_scaling_array, p_projected.x[:, 0], dtype='float')
            key_point_to_zonotope_map[str(p_projected.x[:, 0])]=[p]
        elif p.__name__=='zonotope':
            p_projected = project_zonotope(p, dim=[0, 1], mode='full')
            scaled_key_points[i*(1+2**key_vertex_count),:] = np.multiply(distance_scaling_array, p_projected.x[:, 0], dtype='float')
            key_point_to_zonotope_map[str(p_projected.x[:, 0])]=[p]
            other_key_points = get_k_random_edge_points_in_zonotope_OverR3T(p, generator_idx_list[i], N=key_vertex_count) # key_vertex_count: number of keypoints for zonotope
            # other_key_points = get_k_random_edge_points_in_zonotope(p, generator_idx_list[i], key_vertex_count) # key_vertex_count: number of keypoints for zonotope
            scaled_other_key_points = np.multiply(distance_scaling_array, other_key_points, dtype='float')
            scaled_key_points[i * (2 ** key_vertex_count + 1) + 1:(i + 1) * (2 ** key_vertex_count + 1),
            :] = scaled_other_key_points
            for kp in other_key_points:
                print("kp", kp)
                key_point_to_zonotope_map[str(kp)] = [p]
        else:
            raise NotImplementedError
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
