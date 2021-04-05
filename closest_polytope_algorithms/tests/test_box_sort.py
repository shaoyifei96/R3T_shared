import random
import unittest

from bounding_box.box import *
from bounding_box.box_tree import *
from visualization.visualize import *


class BoxToBoxDistanceTestCase(unittest.TestCase):
    def test_distance_to_box(self):
        x = [(0,0,0),(3,3,3)]
        box1 = AABB(x)
        q1 = AABB([(1,1,1),(3,3,3)])
        self.assertTrue(box_to_box_distance(q1, box1)==0)
        q2 = AABB([(8,8,8),(3,3,3)])
        self.assertTrue(box_to_box_distance(q2,box1)==0)
        q3 = AABB([(2,2,2),(8,8,8)])
        self.assertTrue(box_to_box_distance(q3, box1)==0)
        q4 = AABB([(8,8,8),(4,4,4)])
        self.assertTrue(box_to_box_distance(q4, box1) == 3**0.5)
        q5 = AABB([(4,0,0),(7,3,3)])
        self.assertTrue(box_to_box_distance(q5, box1) == 1)

class BoxNodeTestCase(unittest.TestCase):
    def test_construct_box_node(self):
        x = [(1,1),(3,3)]
        box1 = AABB(x)
        y = [(2,2),(4,4)]
        box2 = AABB(y)
        z = [(3,3),(5,5)]
        box3 = AABB(z)

        bn1 = BoxNode(box1)
        self.assertTrue(bn1.in_this_box(box2))

class BoxTreeTestCase(unittest.TestCase):
    def test_construct_box_tree(self):
        box_list = []
        box_node_list = []
        for i in range(10):
            xs = random.sample(list(range(100)), 2)
            ys = random.sample(list(range(100)),2)
            u = (xs[0],ys[0])
            v = (xs[1],ys[1])
            box = AABB([u,v])
            box_list.append(box)
            box_node_list.append(BoxNode(box))
        root = binary_split(box_node_list)
        # print('root',root)
        # print(box_node_list)

    def test_closest_box(self):
        box_list = []
        box_node_list = []
        for i in range(30):
            centroid = np.asarray(random.sample(list(range(-80,80)), 2))
            edges = np.asarray(random.sample(list(range(1,40)),2))
            box = AABB_centroid_edge(centroid,edges)
            box_list.append(box)
            box_node_list.append(BoxNode(box))

        overlapping_box_list = []
        closest_distance = np.inf
        root = binary_split(box_node_list)
        xs = random.sample(list(range(-80,80)), 2)
        ys = random.sample(list(range(-80,80)), 2)
        u = (xs[0], ys[0])
        v = (xs[1], ys[1])
        test_box = AABB([u, v])
        print(('test bounding_box: ', test_box))
        root.evaluate_node(test_box,overlapping_box_list)
        print(('overlaps with', overlapping_box_list))
        for box in box_list:
            #FIXME: slow implementation
            if box in overlapping_box_list:
                # print(bounding_box)
                # print(box_to_box_distance(test_box, bounding_box))
                self.assertTrue(box_to_box_distance(test_box,box)==0)
            else:
                self.assertTrue(box_to_box_distance(test_box, box)>0)
        axlim = [-100,100]
        fig, ax = visualize_box_nodes(box_node_list,xlim = axlim,ylim=axlim,alpha =0.2,linewidth=0.5)
        fig, ax = visualize_boxes(overlapping_box_list,fig=fig,ax=ax,alpha =1,linewidth=2)
        fig, ax = visualize_boxes([test_box], fig=fig, ax=ax, alpha=1,fill=True,linewidth=0.5)

        plt.show()

if __name__=='__main__':
    unittest.main()