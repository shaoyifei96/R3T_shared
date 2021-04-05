import random
import unittest

from bounding_box.box import *
from visualization.visualize import *
from pypolycontain.lib.zonotope import zonotope
from pypolycontain.visualization.visualize_2D import visualize_2D_zonotopes as visZ
import matplotlib.pyplot as plt

class AABBConstructionTestCase(unittest.TestCase):
    def test_dimension(self):
        x = np.random.rand(2,5)
        box = AABB([x[0,:], x[1,:]])
        self.assertEqual(box.dimension,5)

    def test_vertices(self):
        x = [(3,2,6),(1,5,5)]
        box = AABB(x)
        y = [(1,2,5),(3,5,6)]
        answer = AABB(y)
        self.assertEqual(box.dimension,3)
        self.assertEqual(box.u.any(),answer.u.any())
        self.assertEqual(box.v.any(),answer.v.any())

    def test_equality(self):
        x = [(3,2,6),(1,5,5)]
        box = AABB(x)
        y = [(1,2,5),(3,5,9)]
        b2 = AABB(y)
        b3 = AABB(x)
        self.assertNotEqual(box,b2)
        self.assertEqual(box,b3)

    def test_centroid_construction(self):
        x = np.asarray([1,1,1])
        edges = np.asarray([2,2,2])
        answer = AABB_centroid_edge(x, edges)
        self.assertEqual(answer, AABB([(0,0,0),(2,2,2)]))

    def test_hash(self):
        x = [(3,2,6),(1,5,5)]
        box = AABB(x)
        y = [(1,7,5),(3,5,6)]
        b2 = AABB(y)
        b3 = AABB(x)
        self.assertEqual(hash(box),hash(b3))
        self.assertNotEqual(hash(box),hash(b2))

class AABBCollisionTestCase(unittest.TestCase):
    def test_collision_vertex_overlap(self):
        x = [(3,5,6),(1,5,5)]
        box1 = AABB(x)
        y = [(1,2,5),(3,5,6)]
        box2 = AABB(y)
        self.assertTrue(box1.overlaps(box2))

    def test_collision_contain(self):
        x = [(0,0),(5,5)]
        box1 = AABB(x)
        y = [(1,2),(3,4)]
        box2 = AABB(y)
        self.assertTrue(box1.overlaps(box2))

    def test_collision_overlap(self):
        x = [(0,0),(5,5)]
        box1 = AABB(x)
        y = [(1,2),(9,7)]
        box2 = AABB(y)
        self.assertTrue(box1.overlaps(box2))

    def test_collision_no_overlap(self):
        x = [(5,0),(0,5)]
        box1 = AABB(x)
        y = [(9,10),(-9,6)]
        box2 = AABB(y)
        self.assertFalse(box1.overlaps(box2))

class PointToBoxDistanceTestCase(unittest.TestCase):
    def test_distance_to_box(self):
        x = [(0,0,0),(3,3,3)]
        box1 = AABB(x)
        q1 = (1,1,1)
        self.assertTrue(point_to_box_distance(q1, box1)==0)
        q2 = (0,0,3)
        self.assertTrue(point_to_box_distance(q2,box1)==0)
        q3 = (0,0,4)
        self.assertTrue(point_to_box_distance(q3, box1)==1)
        q4 = (0,-4,0)
        self.assertTrue(point_to_box_distance(q4, box1) == 4)
        q5 = (4, 0, 4)
        self.assertTrue(point_to_box_distance(q5, box1) == 2**0.5)

class ZonotopeToAABBTestCase(unittest.TestCase):
    def test_zonotope_to_AABB(self):
        G_l = np.array([[1, 0, 0, 3], [0, 1, 2, -1]]) * 0.8
        G_r = np.array([[1, 0, 1, 1, 2, -2], [0, 1, 1, -1, 5, 2]]) * 1
        x_l = np.array([0, 1]).reshape(2, 1)
        x_r = np.array([1, 0]).reshape(2, 1)
        zono_l = zonotope(x_l, G_l)
        zono_r = zonotope(x_r, G_r)
        AABB_r = zonotope_to_box(zono_r)
        AABB_l = zonotope_to_box(zono_l)
        fix, ax = visZ([zono_r,zono_l], title="")
        visualize_boxes([AABB_r,AABB_l], ax= ax)
        plt.show()

if __name__=='__main__':
    unittest.main()