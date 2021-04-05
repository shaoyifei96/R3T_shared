import numpy as np
from scipy.spatial import cKDTree as KDTree
from pypolycontain.lib.zonotope import zonotope
from collections import deque
from pypolycontain.lib.AH_polytope import AH_polytope,to_AH_polytope
from pypolycontain.lib.operations import distance_point_polytope
from pypolycontain.lib.polytope import polytope
from pypolycontain.lib.containment_encodings import subset_generic,constraints_AB_eq_CD,add_Var_matrix
from pypolycontain.utils.random_polytope_generator import get_k_random_edge_points_in_zonotope
from gurobipy import Model, GRB, QuadExpr

import itertools
from multiprocessing import Pool
from timeit import default_timer


def set_polytope_pair_distance(arguments):
    key_points, key_point_to_polytope_map, polytope_index, key_point_index = arguments
    key_point = key_points[key_point_index]
    key_point_string = str(key_point)
    polytope = key_point_to_polytope_map[key_point_string]['polytopes'][polytope_index]
    return distance_point_polytope(to_AH_polytope(polytope), key_point, ball='l2')[0]

class VoronoiClosestPolytope:
    def __init__(self, polytopes, key_vertices_count=0, process_count=8, max_number_key_points = None):
        '''
        Compute the closest polytope using Voronoi cells
        :param polytopes:
        '''
        self.init_start_time = default_timer()
        self.section_start_time = self.init_start_time
        self.polytopes = np.asarray(polytopes, dtype='object')
        self.type = self.polytopes[0].type
        self.process_count = process_count
        self.key_vertices_count = key_vertices_count
        if self.type == 'AH_polytope':
            self.dim = self.polytopes[0].t.shape[0]
        elif self.type == 'zonotope':
            self.dim =self.polytopes[0].x.shape[0]
        else:
            raise NotImplementedError
        if self.key_vertices_count>0:
            self.key_points = np.zeros([len(self.polytopes) * (1 + 2 ** self.key_vertices_count), self.dim])
        else:
            self.key_points = np.zeros([len(self.polytopes), self.dim])
        for i, z in enumerate(polytopes):
            if self.type == 'AH_polytope':
                if self.key_vertices_count>0:
                    raise NotImplementedError
                else:
                    self.key_points[i, :] = self.polytopes[i].t[:, 0]
            elif self.type == 'zonotope':
                if self.key_vertices_count>0:
                    self.key_points[i * (2 ** self.key_vertices_count + 1), :] = self.polytopes[i].x[:, 0]
                    self.key_points[i*(2 ** self.key_vertices_count + 1)+1:(i + 1) * (2 ** self.key_vertices_count + 1), :] = get_k_random_edge_points_in_zonotope(self.polytopes[i], self.key_vertices_count)
                else:
                    self.key_points[i, :] = self.polytopes[i].x[:, 0]
            else:
                raise NotImplementedError
        if max_number_key_points:
            # sample the key points
            n = self.key_points.shape[0]
            chosen_key_points = np.random.choice(n, size=min(n, max_number_key_points), replace=False)
            self.key_points = self.key_points[chosen_key_points, :]
            # print(self.key_points.shape)
        self.key_point_to_polytope_map = dict()  # stores the potential closest polytopes associated with each Voronoi (centroid)
        for key_point in self.key_points:
            ds = np.zeros(self.polytopes.shape[0])
            self.key_point_to_polytope_map[str(key_point)] = np.rec.fromarrays([self.polytopes, ds], names=('polytopes', 'distances'))

        self.build_cell_polytope_map_default()

        #build kd-tree for centroids
        self.key_point_tree = KDTree(self.key_points)
        print(('Completed precomputation in %f seconds' % (default_timer() - self.init_start_time)))

    def build_cell_polytope_map_default(self):
        polytope_key_point_indices = np.array(np.meshgrid(np.arange(self.polytopes.shape[0]), np.arange(self.key_points.shape[0]))).T.reshape(-1, 2)
        arguments = []
        for i in polytope_key_point_indices:
            arguments.append((self.key_points, self.key_point_to_polytope_map, i[0], i[1]))
        p = Pool(self.process_count)
        pca = p.map(set_polytope_pair_distance, arguments)
        polytope_key_point_arrays=np.asarray(pca).reshape((self.polytopes.shape[0]), self.key_points.shape[0])
        # print(polytope_centroid_arrays)
        # compute pairwise distances of the centroids and the polytopes
        #fixme
        for key_point_index, key_point in enumerate(self.key_points):
            key_point_string = str(key_point)
            for polytope_index, polytope in enumerate(self.key_point_to_polytope_map[key_point_string]['polytopes']):
                self.key_point_to_polytope_map[str(key_point)].distances[polytope_index] = polytope_key_point_arrays[polytope_index, key_point_index]
                # print(polytope_key_point_arrays[polytope_index, key_point_index])
            self.key_point_to_polytope_map[key_point_string].sort(order='distances')
            # print(self.centroid_to_polytope_map[centroid_string])

    def find_closest_polytope(self, query_point, return_intermediate_info = False):
        #find the closest centroid
        d,i = self.key_point_tree.query(query_point)
        closest_key_point = self.key_point_tree.data[i]
        # print('closest key point', closest_key_point)
        closest_key_point_polytope = self.key_point_to_polytope_map[str(closest_key_point)]['polytopes'][0]
        # print('closest polytope centroid' + str(closest_key_point_polytope.x))
        dist_query_centroid_polytope = distance_point_polytope(closest_key_point_polytope, query_point, ball='l2')[0]
        dist_query_key_point = np.linalg.norm(query_point-closest_key_point)
        # print(dist_query_key_point, dist_query_centroid_polytope)
        cutoff_index = np.searchsorted(self.key_point_to_polytope_map[str(closest_key_point)].distances, dist_query_key_point + dist_query_centroid_polytope)
        # print(cutoff_index)
        # print(self.key_point_to_polytope_map[str(closest_key_point)]['distances'][0:cutoff_index])
        # print(self.key_point_to_polytope_map[str(closest_key_point)]['distances'][cutoff_index:])
        # print('dqc',dist_query_key_point)
        # print(self.centroid_to_polytope_map[str(closest_key_point)].distances)
        closest_polytope_candidates = self.key_point_to_polytope_map[str(closest_key_point)].polytopes[0:cutoff_index]
        # print(closest_polytope_candidates)
        best_polytope = None
        best_distance = np.inf
        for polytope in closest_polytope_candidates:
            if best_distance < 1e-9:
                break
            dist = distance_point_polytope(polytope, query_point, ball='l2')[0]
            if best_distance>dist:
                best_distance = dist
                best_polytope = polytope
        # print('best distance', best_distance)
        if return_intermediate_info:
            return best_polytope, best_distance, closest_polytope_candidates
        return best_polytope