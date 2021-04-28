import numpy as np
from pypolycontain.lib.zonotope import zonotope, zonotope_directed_distance
from pypolycontain.visualization.visualize_2D import visualize_2D_zonotopes as visZ
from pypolycontain.lib.zonotope import zonotope_inside, zonotope_distance_point

# AABB collision check
from closest_polytope_algorithms.bounding_box.box import point_in_box
from pypolycontain.lib.operations import to_AH_polytope
from closest_polytope_algorithms.bounding_box.box import AH_polytope_to_box

from matplotlib.pyplot import show

import scipy.io

import time

# --------------------- Global Variables ---------------------
# Simon: Why is this here? makes my code break... why need global
# mat = scipy.io.loadmat(
#     "/home/yingxue/R3T_shared/r3t/overapproximate_with_slice/test_zono.mat"
# )

# --------------------- helper functions ---------------------
# slicing
                    #   3x1                     
def zonotope_slice(z, generator_idx=[1, 2, 3], slice_dim=[2, 3, 4], slice_value=[0, 0.2, 0.0004]):
                                                #comeon!! this needs to subtract 1, fuk
    '''
    generator_idx: always len 3
    '''
    generator_idx = np.array(generator_idx).squeeze()
    slice_G = z.G[slice_dim, generator_idx]

    # print("G",z.G.shape)
    # print("slice G",slice_G)
    # print("slice_dim", slice_dim)
    # # print("generator_idx", generator_idx)
    # print("slice_G", slice_G.shape)
    # print("z.x[slice_dim]", z.x[slice_dim].shape)
    # exit()                          
    # gen_idx ={t0_gen:593,t0_dot:595,k:594}
    #                                593    594    595
    #          c         G 6x600     t0_gen  k      t0_dot
    # t                              1       2      1
    # t_dot                          2       4      5
    # t0                             0.5     0      0
    # t_dot_0                        0       0      0.5
    # k                              0       0.1    0
    # t                              0  
    newG = np.delete(z.G, generator_idx, 1)

    slice_lambda = np.divide((slice_value - z.x[slice_dim].reshape(-1, )), slice_G)
    if len(slice_dim) == 1:
        sliceable_G = z.G[:, generator_idx].reshape((-1,1))
    else:
        sliceable_G = z.G[:, generator_idx]

    # print("newG", newG.shape)
    # print("gen",generator_idx)
    # print("sliceable_G", sliceable_G.shape)
    # print("slice_lambda", np.reshape(slice_lambda,(-1,1)).shape)
    newc = np.matmul(sliceable_G, np.reshape(slice_lambda,(-1,1))) + z.x.reshape((-1, 1))
    # print("newc", newc,newc.shape)
    # newc = newc.reshape((-1, 1))
    # exit()

    return zonotope(newc, newG, color="green")


def project_zonotope(Z, dim, mode ='full'):
    '''
    dim : list, e.g. [0, 1]
    mode: center/full
    '''
    if mode == 'full':
        return zonotope(Z.x[dim], Z.G[dim, :]); 
    else:
        return Z.x[dim]


def convert_obs_to_zonotope(c,theta_len,theta_dot_length):
    '''
    Assume 2D obstacles in dim 0 = theta and dim 1 = theta_dot space.
    c = 2x1 G[:,0] = [theta_len/2  0] G[:,1] = [0 theta_dot_length/2]
    or be creative with your G, doesn't have to have two dim and they can be related to
    make osbtacle of any shape in state space, just know the speed is gonna suffer if you have more generators
    '''
    newc = c.reshape((-1, 1))
    newG = np.array([[theta_len/2, 0], [0, theta_dot_length/2]])
    return zonotope(newc, newG, color="red")


def check_zonotope_collision(zono_list, gen_idx_list, k, state_initial, Z_obs_list=None, max_time_index = 1):
    '''
    Check complete_reachable_set(), get list of reachable sets for each t using k from the dict
    '''

    # Z_obs_list = []
    # G_l = np.array([[0.1, 0], [0, 0.3]])*0.5
    # x_l = np.array([3., 0.]).reshape(2, 1)
    # # x_l = np.array([1., 4.]).reshape(2, 1)
    # z_obs = zonotope(x_l, G_l)
    # Z_obs_list.append(z_obs)
    # print(zono_list, gen_idx_list, k, state_initial, Z_obs_list)

    # check last one first, largest, more likely to intersect with things
    # for zono_idx in reversed(range(len(zono_list))):
    # print("max_time_index",max_time_index)
    for zono_idx in reversed(range(max_time_index+1)):
        # print(zono_list[zono_idx], gen_idx_list[zono_idx], np.append(state_initial,k))
        generator_idx = gen_idx_list[zono_idx]
        generator_idx = generator_idx[2] - int(generator_idx[2]>generator_idx[0]) - int(generator_idx[2]>generator_idx[1])
        zono = zonotope_slice(zono_list[zono_idx], generator_idx[-1], slice_dim=[4], slice_value=k)
        for Z_obs in Z_obs_list:
            if check_zono_contain(zono, Z_obs):
                return True
            
            # Box-to-box collision
            # Current code is incorrect, it only check the center of zonotope with obs boxes
            # obs_p = to_AH_polytope(Z_obs)
            # obs_box = AH_polytope_to_box(obs_p, return_AABB = True)
            # if point_in_box(zono.x, obs_box):
            #     # collided, Just discard, same as plan_collision_free_path_in_set()
            #     return True

    return False


def check_zono_contain(Z, Z_obs):
    '''
    assume obstacle exists in the first two dim
    '''
    # print((Z_obs.x.shape),(Z.x[0:2,0].shape))
    new_c = Z_obs.x - Z.x[0:2,0].reshape((2,1))
    shrinked_G = Z.G[0:2,:]

    idx = np.argwhere(np.all(abs(shrinked_G[..., :]) < 1e-5, axis=0))
    shrinked_G = np.delete(shrinked_G, idx, axis=1) #delete all zeros generators
    new_G = np.concatenate((Z_obs.G,shrinked_G),axis=1) # there may be a lot of zeros since just using the first few dims of the G
    buffered_Z = zonotope(new_c, new_G, color = "red")
    # print("newc newG",new_c,new_G)

    # distance  = zonotope_distance_point(buffered_Z,np.zeros((2,1)) )[0]
    # # print("distance to zonotope (Collision)",distance)
    # if (distance<0.2):
    #     return True
    # else:
    #     return False

    return zonotope_inside(buffered_Z, np.zeros((2,1)))


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
