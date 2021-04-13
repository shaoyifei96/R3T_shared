import numpy as np
from pypolycontain.lib.zonotope import zonotope, zonotope_directed_distance
from pypolycontain.visualization.visualize_2D import visualize_2D_zonotopes as visZ
from matplotlib.pyplot import show

import scipy.io

# --------------------- Global Variables ---------------------

mat = scipy.io.loadmat(
    "/home/yingxue/R3T_shared/r3t/overapproximate_with_slice/test_zono.mat"
)

# --------------------- helper functions ---------------------
# slicing

def zonotope_slice(z, generator_idx=[1, 2, 3], slice_dim=[3, 4, 5], slice_value=[0, 0.2, 0.0004]):
    '''
    generator_idx: always len 3
    '''

    slice_G = z.G[slice_dim, generator_idx]

    slice_lambda = np.linalg.solve(slice_G, slice_value - z.x[slice_dim])
    newG = np.delete(z.G, generator_idx, 1)
    newc = np.matmul(z.G[:, generator_idx].squeeze(), slice_lambda) + z.x
    newc = newc.reshape((-1, 1))

    return zonotope(newc, newG, color="red")
