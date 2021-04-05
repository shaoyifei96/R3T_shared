import numpy as np
from voronoi.voronoi import VoronoiClosestPolytope
from pypolycontain.lib.zonotope import zonotope
from visualization.visualize import *
from pypolycontain.visualization.visualize_2D import visualize_2D_zonotopes as visZ

def test_zonotope_to_voronoi():
    zonotope_count = 10
    zonotopes = []
    centroid_range = zonotope_count*4
    generator_range = 3
    for i in range(zonotope_count):
        m = np.random.random_integers(4, 10)
        G = (np.random.rand(2, m) - 0.5) * generator_range
        x = (np.random.rand(2, 1) - 0.5) * centroid_range
        zonotopes.append(zonotope(x, G))

    vcp = VoronoiClosestPolytope(zonotopes)

if __name__=='__main__':
    test_zonotope_to_voronoi()