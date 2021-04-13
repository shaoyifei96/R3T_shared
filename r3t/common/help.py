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


def project_zonotope(Z, dim, mode ='full'):
    '''
    dim : list
    mode: center/full
    '''
    if mode == 'full':
        return zonotope(Z.x[dim], Z.G[dim, :]); 
    else:
        return Z.x[dim]




# def get_k_random_edge_points_in_zonotope_OverR3T(zonotope, k0=[0], kf=[1], N=5):
#     ''' slice it for a particular parametere value and then find end point of zonotope.
#     keypoints are center of the zonotopes.

#     Steps: 

#     @ params:
#     ---
#     Z:      is the last zonotope
#     k0:     lowest value of parameter
#     kf:     highest value of parameter
#     N:      number of keypoints to consider
#     '''

#     keypoints = []
#     slice_dim=[4]
#     generator_idx=[]

#     ki_list = np.linspace(k0, kf, N)
#     # print(ki_list)
#     slice_lists = np.vstack(np.meshgrid(*ki_list.T)).reshape(len(k0),-1).T
#     # print(slice_lists)

#     for slice_value in slice_lists:
#         Z_sliced = zonotope_slice(zonotope, slice_dim=slice_dim, slice_value=slice_value)
#         # z, generator_idx=[305, 306, 307], slice_dim=[3, 4, 5], slice_value=[0, 0.2, 0.0004]
#         slice_G = Z_sliced.G[slice_dim, slice_idx]
