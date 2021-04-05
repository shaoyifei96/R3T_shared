# -*- coding: utf-8 -*-
'''

@author: wualbert
'''

import random
import unittest
import matplotlib
from time import time
from bounding_box.polytope_tree import *
from visualization.visualize import *
from pypolycontain.visualization.visualize_2D import visualize_2D_zonotopes as visZ
from pypolycontain.utils.random_polytope_generator import *
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

matplotlib.rcParams['font.family'] = "Times New Roman"
matplotlib.rcParams.update({'font.size': 14})

class CentroidKDTreeTestCase(unittest.TestCase):
    def test_kd_construction(self):
        pass
    def test_query(self):
        pass

class ZonotopeTreeTestCase(unittest.TestCase):
    def test_few_zonotopes(self):
        G_l = np.array([[1, 0, 0, 3], [0, 1, 2, -1]]) * 0.8
        G_r = np.array([[1, 0, 1, 1, 2, -2], [0, 1, 1, -1, 5, 2]]) * 1
        x_l = np.array([0., 1.]).reshape(2, 1)
        x_r = np.array([5., 0.]).reshape(2, 1)
        zono_l = zonotope(x_l, G_l)
        zono_r = zonotope(x_r, G_r)
        zt = PolytopeTree([zono_l, zono_r])
        query_point = np.asarray([0,-5])
        np.reshape(query_point,(query_point.shape[0],1))
        closest_zonotope, best_distance, evaluated_zonotopes, query_point = zt.find_closest_polytopes(np.asarray(query_point), return_intermediate_info=True)
        # print(closest_zonotope)
        fig, ax = visZ([zono_r,zono_l], title="", alpha=0.2)
        plt.scatter(query_point[0],query_point[1])
        fig, ax = visZ(closest_zonotope, title="",fig=fig,ax=ax,alpha=0.75)
        # fig, ax = visualize_box_nodes(zt,fig=fig,ax=ax,alpha =0.4)
        print(('Closest Zonotope: ', closest_zonotope))
        plt.show()

    def test_many_zonotopes(self, distance_scaling_array = np.ones(2, dtype='float')):
        zonotope_count = 15
        centroid_range = zonotope_count * 1.5
        # seed = int(time())
        seed = np.random.random_integers(0,10000000,1)  # int(time())
        np.random.seed(seed)
        zonotopes = get_uniform_random_zonotopes(zonotope_count, dim=2, generator_range=zonotope_count * 0.3,
                                              centroid_range=centroid_range, return_type='zonotope', seed=seed)
        zt = PolytopeTree(zonotopes, distance_scaling_array=distance_scaling_array)

        query_point = np.asarray([np.random.random_integers(-centroid_range,centroid_range),
                       np.random.random_integers(-centroid_range,centroid_range)])
        query_point = query_point.reshape(-1,1)
        closest_zonotope, best_distance,evaluated_zonotopes,query_box = zt.find_closest_polytopes(query_point, return_intermediate_info=True)
        print(('Solved %d LP' %len(evaluated_zonotopes)))
        print(('Best distance: ', best_distance))
        print(('Query point: ', query_point))
        ax_lim = np.asarray([-centroid_range,centroid_range,-centroid_range,centroid_range])*1.1

        fig, ax = visZ(zonotopes, title="", alpha=0.2,axis_limit=ax_lim, color='black')
        fig, ax = visualize_boxes([zonotope_to_box(p, return_AABB=False) for p in zt.polytopes],fig=fig,ax=ax,alpha =0.08,linewidth=0.5, facecolor='black')
        fig, ax = visualize_boxes([zonotope_to_box(p, return_AABB=False) for p in evaluated_zonotopes],fig=fig,ax=ax,alpha =0.2,linewidth=0.5, facecolor='red')

        fig, ax = visZ(closest_zonotope, title="",fig=fig,ax=ax,alpha=1,axis_limit=ax_lim, color='blue')
        for vertex in zt.scaled_key_point_tree.data:
            plt.scatter(*np.divide(vertex, distance_scaling_array), facecolor='c', s=2, alpha=1)
        # fig, ax = visualize_boxes(candidate_boxes,fig=fig,ax=ax,alpha =1)
        # print('Candidate boxes: ', candidate_boxes)
        fig, ax = visualize_boxes([query_box], fig=fig, ax=ax, alpha=0.3,
                                  xlim=[-centroid_range,centroid_range],ylim=[-centroid_range,centroid_range], facecolor='cyan')
        # lim = 60
        # ax.set_xlim(-lim,lim)
        # ax.set_ylim(-lim,lim)
        ax.xaxis.set_major_locator(MultipleLocator(15))
        ax.yaxis.set_major_locator(MultipleLocator(15))
        ax.set_title('Nearest Polytope Querying with AABB')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        ax.scatter(query_point[0],query_point[1],s=10,color='k')
        print(('Closest Zonotope: ', closest_zonotope))
        plt.savefig('closest_zonotope.png', dpi=300)
        plt.close()

    def test_scaled_many_zonotopes_repeat(self):
        for i in range(100):
            self.test_many_zonotopes(np.asarray([1000., 1.]))

    def test_many_zonotopes_repeat(self):
        for itr in range(100):
            zonotope_count = 50
            zonotopes = []
            centroid_range = zonotope_count
            generator_range = 3
            for i in range(zonotope_count):
                m = np.random.random_integers(4, 10)
                G = (np.random.rand(2, m) - 0.5) * generator_range
                x = (np.random.rand(2, 1) - 0.5) * centroid_range
                zonotopes.append(zonotope(x, G))
            zt = PolytopeTree_Old(zonotopes)

            query_point = np.asarray([np.random.random_integers(-centroid_range, centroid_range),
                                      np.random.random_integers(-centroid_range, centroid_range)])
            query_point = query_point.reshape(-1, 1)
            closest_zonotope, candidate_boxes, query_box = zt.find_closest_zonotopes(query_point)
            # print('Query point: ', query_point)
            # ax_lim = np.asarray([-centroid_range, centroid_range, -centroid_range, centroid_range]) * 1.1
            # fig, ax = visZ(zonotopes, title="", alpha=0.2, axis_limit=ax_lim)
            # fig, ax = visZ(closest_zonotope, title="", fig=fig, ax=ax, alpha=1, axis_limit=ax_lim)
            # fig, ax = visualize_box_nodes(zt.box_nodes, fig=fig, ax=ax, alpha=0.4, linewidth=0.5)
            # fig, ax = visualize_boxes(candidate_boxes, fig=fig, ax=ax, alpha=1)
            # # print('Candidate boxes: ', candidate_boxes)
            # fig, ax = visualize_boxes([query_box], fig=fig, ax=ax, alpha=0.3, fill=True)
            # plt.scatter(query_point[0], query_point[1], s=20, color='k')
            # print('Closest Zonotope: ', closest_zonotope)
            assert(len(closest_zonotope)>0)
            print(('Completed iteration ',itr))
            # plt.show()

    def test_many_zonotopes_line(self):
        zonotope_count = 50
        zonotopes = []
        centroid_range = zonotope_count
        generator_range = 3
        for i in range(zonotope_count):
            m = np.random.random_integers(4,10)
            G = (np.random.rand(2,m)-0.5)*generator_range*1
            x = np.asarray([(np.random.rand(1)-0.5)*2*centroid_range,
                            np.random.rand(1)-0.5])
            zonotopes.append(zonotope(x,G))
        zt = PolytopeTree_Old(zonotopes)

        query_point = np.asarray([np.random.random_integers(-centroid_range,centroid_range),
                       np.random.rand(1)-0.5])
        query_point = query_point.reshape(-1,1)
        closest_zonotope, candidate_boxes,query_box = zt.find_closest_zonotopes(query_point)
        print(('Query point: ', query_point))
        ax_lim = np.asarray([-centroid_range,centroid_range,-centroid_range,centroid_range])*1.1
        fig, ax = visZ(zonotopes, title="", alpha=0.2,axis_limit=ax_lim)
        fig, ax = visZ(closest_zonotope, title="",fig=fig,ax=ax,alpha=1,axis_limit=ax_lim)
        fig, ax = visualize_box_nodes(zt.box_nodes,fig=fig,ax=ax,alpha =0.4,linewidth=0.5)
        fig, ax = visualize_boxes(candidate_boxes,fig=fig,ax=ax,alpha =1)
        print(('Candidate boxes: ', candidate_boxes))
        fig, ax = visualize_boxes([query_box], fig=fig, ax=ax, alpha=0.3,fill=True)
        plt.scatter(query_point[0],query_point[1],s=20,color='k')
        print(('Closest Zonotope: ', closest_zonotope))
        plt.show()

    def test_with_dataset(self):
        from utils.PWA_Control_utils import polytree_to_zonotope_tree
        import pickle
        with open("zonotope_datasets/inverted_pendulum.pkl", "rb") as f:
            state_tree = pickle.load(f)
        f.close()
        #extract zonotopes
        zonotope_tree = polytree_to_zonotope_tree(state_tree)
        zonotope_count = len(zonotope_tree.zonotopes)

        query_point = np.asarray([(np.random.rand(1) - 0.5)*0.24,
                                  (np.random.rand(1) - 0.5)*2])
        print(query_point)
        query_point = query_point.reshape(-1, 1)
        closest_zonotope, candidate_boxes, query_box = zonotope_tree.find_closest_polytopes(query_point)
        print(('Query point: ', query_point))
        ax_lim = np.asarray([-zonotope_count, zonotope_count, -zonotope_count, zonotope_count]) * 1.1
        fig, ax = visZ(zonotope_tree.zonotopes, title="", alpha=0.2, axis_limit=ax_lim)
        fig, ax = visZ(closest_zonotope, title="", fig=fig, ax=ax, alpha=1, axis_limit=ax_lim)
        fig, ax = visualize_box_nodes(zonotope_tree.box_nodes, fig=fig, ax=ax, alpha=0.4, linewidth=0.5)
        fig, ax = visualize_boxes(candidate_boxes, fig=fig, ax=ax, alpha=1)
        print(('Evaluating %d zonotopes out of %d' %(len(candidate_boxes),len(zonotope_tree.zonotopes))))
        fig, ax = visualize_boxes([query_box], fig=fig, ax=ax, alpha=0.3, fill=True)
        plt.scatter(query_point[0], query_point[1], s=20, color='k')
        print(('Closest Zonotope: ', closest_zonotope))
        plt.show()

    def test_insertion(self):
        zonotope_count = 10
        insert_count = 5
        centroid_range = zonotope_count * 6
        seed = np.random.random_integers(0,10000000,1)  # int(time())
        print(('Seed: ', seed))
        all_zonotopes = get_uniform_random_zonotopes(zonotope_count+insert_count, dim=2, generator_range=zonotope_count * 1,
                                                 centroid_range=centroid_range, return_type='zonotope', seed=seed)
        insert_zonotopes = all_zonotopes[zonotope_count:]
        zonotopes = all_zonotopes[0:zonotope_count]
        zt = PolytopeTree(zonotopes)

        query_point = np.asarray([np.random.random_integers(-centroid_range, centroid_range),
                                  np.random.random_integers(-centroid_range, centroid_range)])
        query_point = query_point.reshape(-1, 1)
        closest_zonotope, best_distance, evaluated_zonotopes, query_box = zt.find_closest_polytopes(query_point,
                                                                                                    return_intermediate_info=True)
        print(('Solved %d LP' % len(evaluated_zonotopes)))
        print(('Best distance: ', best_distance))
        print(('Query point: ', query_point))
        ax_lim = np.asarray([-centroid_range, centroid_range, -centroid_range, centroid_range]) * 1.1
        fig, ax = visZ(zonotopes, title="", alpha=0.2, axis_limit=ax_lim)
        fig, ax = visZ(closest_zonotope, title="", fig=fig, ax=ax, alpha=1, axis_limit=ax_lim)
        # fig, ax = visualize_box_nodes(zt.box_nodes,fig=fig,ax=ax,alpha =0.4,linewidth=0.5)
        for vertex in zt.scaled_key_point_tree.data:
            plt.scatter(vertex[0], vertex[1], facecolor='c', s=2, alpha=1)

        # fig, ax = visualize_boxes(candidate_boxes,fig=fig,ax=ax,alpha =1)
        # print('Candidate boxes: ', candidate_boxes)
        fig, ax = visualize_boxes([query_box], fig=fig, ax=ax, alpha=0.1,
                                  xlim=[-centroid_range, centroid_range], ylim=[-centroid_range, centroid_range])
        # ax.set_xlim(-centroid_range,centroid_range)
        # ax.set_ylim(-centroid_range,centroid_range)

        ax.scatter(query_point[0], query_point[1], s=10, color='k')
        print(('Closest Zonotope: ', closest_zonotope))
        plt.show()

        #insert
        zt.insert(insert_zonotopes)
        closest_zonotope, best_distance, evaluated_zonotopes, query_box = zt.find_closest_polytopes(query_point,
                                                                                                    return_intermediate_info=True)
        print(('Solved %d LP' % len(evaluated_zonotopes)))
        print(('Best distance: ', best_distance))
        print(('Query point: ', query_point))
        ax_lim = np.asarray([-centroid_range, centroid_range, -centroid_range, centroid_range]) * 1.1
        fig, ax = visZ(all_zonotopes, title="", alpha=0.2, axis_limit=ax_lim)
        fig, ax = visZ(closest_zonotope, title="", fig=fig, ax=ax, alpha=1, axis_limit=ax_lim)
        # fig, ax = visualize_box_nodes(zt.box_nodes,fig=fig,ax=ax,alpha =0.4,linewidth=0.5)
        for vertex in zt.scaled_key_point_tree.data:
            plt.scatter(vertex[0], vertex[1], facecolor='c', s=2, alpha=1)
        print((len(zt.scaled_key_point_tree.data)))
        # fig, ax = visualize_boxes(candidate_boxes,fig=fig,ax=ax,alpha =1)
        # print('Candidate boxes: ', candidate_boxes)
        fig, ax = visualize_boxes([query_box], fig=fig, ax=ax, alpha=0.1,
                                  xlim=[-centroid_range, centroid_range], ylim=[-centroid_range, centroid_range])
        # ax.set_xlim(-centroid_range,centroid_range)
        # ax.set_ylim(-centroid_range,centroid_range)

        ax.scatter(query_point[0], query_point[1], s=10, color='k')
        print(('Closest Zonotope: ', closest_zonotope))
        plt.show()


    # def test_zonotope_boxes(self):
    #     '''
    #     for debugging
    #     '''
    #     zonotope_count = 50
    #     zonotopes = []
    #     centroid_range = zonotope_count
    #     generator_range = 3
    #     for i in range(zonotope_count):
    #         m = np.random.random_integers(4, 10)
    #         G = (np.random.rand(2, m) - 0.5) * generator_range
    #         x = (np.random.rand(2, 1) - 0.5) * centroid_range
    #         zonotopes.append(zonotope(x, G))
    #     zt = ZonotopeTree(zonotopes)
    #     bn = zt.box_nodes
    #     query_point = np.asarray([np.random.random_integers(-centroid_range,centroid_range),
    #                    np.random.random_integers(-centroid_range,centroid_range)])
    #     query_point = query_point.reshape(-1,1)
    #
    #     ax_lim = np.asarray([-centroid_range,centroid_range,-centroid_range,centroid_range])*1.1
    #     closest_zonotope,_c,_d = zt.find_closest_zonotopes(np.asarray(query_point))
    #     fig, ax = visZ(closest_zonotope, title="",alpha=1,axis_limit=ax_lim)
    #     fig, ax = visualize_box_nodes(zt.box_nodes,fig=fig,ax=ax,alpha =0.4,linewidth=0.5)
    #     #brute force calculation
    #     bfc_centroid = None
    #     bfc_distance = np.inf
    #     for z in zt.zonotopes:
    #         plt.scatter(z.x[0], z.x[1], s=20, color='b')
    #         vec_diff = np.subtract(np.ndarray.flatten(z.x),\
    #                                np.ndarray.flatten(query_point))
    #         dis = np.linalg.norm(vec_diff)
    #         if bfc_distance>dis:
    #             bfc_centroid=z.x
    #             bfc_distance=dis
    #     plt.scatter(bfc_centroid[0], bfc_centroid[1], s=20, color='g')
    #
    #     # plt.scatter(closest_zonotope[0].x[0], closest_zonotope[0].x[1], s=20, color='r')
    #
    #     plt.scatter(query_point[0],query_point[1],s=20,color='k')
    #
    #     # edge_length = closest_zonotope[0].x
    #     # query_box = closest_zonotope
    #
    #
    #     # query_point = np.asarray([np.random.random_integers(-centroid_range,centroid_range),
    #     #                np.random.random_integers(-centroid_range,centroid_range)])
    #     # query_point = query_point.reshape(-1,1)
    #     # closest_zonotope, candidate_boxes,query_box = zt.find_closest_zonotopes(query_point)
    #     # print('Query point: ', query_point)
    #     # ax_lim = np.asarray([-centroid_range,centroid_range,-ce
    #     # ntroid_range,centroid_range])*1.1
    #     # fig, ax = visZ(zonotopes, title="", alpha=0.2,axis_limit=ax_lim)
    #     # fig, ax = visZ(closest_zonotope, title="",fig=fig,ax=ax,alpha=1,axis_limit=ax_lim)
    #     # fig, ax = visualize_box_nodes(zt.box_nodes,fig=fig,ax=ax,alpha =0.4,linewidth=0.5)
    #     # fig, ax = visualize_boxes(candidate_boxes,fig=fig,ax=ax,alpha =1)
    #     # print('Candidate boxes: ', candidate_boxes)
    #     # fig, ax = visualize_boxes([query_box], fig=fig, ax=ax, alpha=0.3,fill=True)
    #     # plt.scatter(query_point[0],query_point[1],s=20,color='k')
    #     # print('Closest Zonotope: ', closest_zonotope)
    #     plt.show()

if __name__=='__main__':
    unittest.main()