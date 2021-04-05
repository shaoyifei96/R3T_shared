import random
import unittest

from bounding_box.box import *
from visualization.visualize import *
import matplotlib.pyplot as plt

class AABBVisualizationTestCase(unittest.TestCase):
    def test_AABB_visulaization(self):
        x = [(0,0,0),(3,3,3)]
        box1 = AABB(x)
        visualize_boxes([box1])
        plt.show()