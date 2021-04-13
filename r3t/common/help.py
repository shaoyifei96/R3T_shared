import numpy as np
from pypolycontain.lib.zonotope import zonotope, zonotope_directed_distance
from pypolycontain.visualization.visualize_2D import visualize_2D_zonotopes as visZ
from matplotlib.pyplot import show

import scipy.io



# --------------------- helper functions ---------------------
# slicing

def zonotope_slice(
    z, generator_idx=[305, 306, 307], slice_dim=[3, 4, 5], slice_value=[0, 0.2, 0.0004]
):
    '''
    generator_idx: always len 3
    '''

    generator_idx = generator_idx[slice_dim - 3]
    slice_G = z.G[slice_dim, generator_idx]

    slice_lambda = np.linalg.solve(slice_G, slice_value - z.x[slice_dim])
    newc = np.matmul(z.G[:, generator_idx].squeeze(), slice_lambda) + z.x

    return zonotope(newc, newG, color="red")
