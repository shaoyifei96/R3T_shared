import numpy as np
from pypolycontain.lib.zonotope import zonotope,zonotope_directed_distance
from pypolycontain.visualization.visualize_2D import visualize_2D_zonotopes as visZ
from matplotlib.pyplot import show
from r3t.common.help import 

import scipy.io# for matab import, zonotope array saved as cell arrary of matrix, dim x num_generator +1, MATLAB: c = Z(:,1) G = Z(:,2:end)
def zonotope_slice_345(z,slice_idx,slice_value):
    slice_dim = np.array([2,3,4])#slice number -1 since python
    slice_G = z.G[slice_dim, slice_idx]
    # print(slice_G.shape)
    # print(z.G[:, slice_idx].shape)

    slice_lambda = np.linalg.solve(slice_G, slice_value - z.x[slice_dim])#MATLAB: slice_G\(slice_pt - slice_c);
    # print(slice_lambda.shape)
    newG = np.delete(z.G, slice_idx, 1)
    newc =  np.matmul(z.G[:, slice_idx].squeeze(),slice_lambda) + z.x
    print(f"slide_idx:{slice_idx}")
    print(f"New center:{newc}") 

def zonotope_slice_34(z,slice_idx,slice_value):
    slice_dim = np.array([2,3])#slice number -1 since python
    slice_G = z.G[slice_dim, slice_idx]

    slice_lambda = np.linalg.solve(slice_G, slice_value - z.x[slice_dim])
    newc =  np.matmul(z.G[:, slice_idx].squeeze(),slice_lambda) + z.x
    print(f"slide_idx:{slice_idx}")
    print(f"New center:{newc}") 
    
    return zonotope(newc,newG,color="red")

# generator_idx: always 3
def zonotope_slice(z,generator_idx = [305,306,307],slice_dim=[3,4,5],slice_value = [0,0.2,0.0004]):
    
    generator_idx = generator_idx[slice_dim-3]
    slice_G = z.G[slice_dim, generator_idx]

    slice_lambda = np.linalg.solve(slice_G, slice_value - z.x[slice_dim])
    newc =  np.matmul(z.G[:, generator_idx].squeeze(),slice_lambda) + z.x

    return zonotope(newc,newG,color="red")

# 3D mesh with one parameter range and two initial condition range
mat = scipy.io.loadmat('/home/simon/Documents/MP_backup/motion_planning_598/final/R3T_shared/r3t/overapproximate_with_slice/test_zono.mat')
            #var     #=0 #time 
print("printing loaded data stats")
print(f"mat info \n :{mat['info_FRS']}")
print(f"mat save \n :{mat['save_FRS'].shape}")
print(f"mat['save_FRS'][0][1] .shape: \n {mat['save_FRS'][0][1] .shape}" )
print(f"center: mat['save_FRS'][0] [10] [:,0]:\n {mat['save_FRS'][0] [10] [:,0]}")#center
print(f"Generator as a matrix: mat['save_FRS'][0] [10] [:,1:].shape: \n {mat['save_FRS'][0] [10] [:,1:].shape}" )#Generator as a matrix
print(f"Generators to slice for 10th time interval \n mat['info_FRS'][0] [10]:{mat['info_FRS'][0] [10]}" ) #what are the generators to slice for the 10th time interval
#info contain info about which generator to slice during online for each zonotope, in order [theta; thetadot; k]

G_1=mat['save_FRS'][0] [10] [:,1:] # all other are generators
# print(G_1.shape)
x_1=mat['save_FRS'][0] [10] [:,0] #center

z1=zonotope(x_1,G_1,color="green")
slice_value  = np.array([0.1324, -0.4025, 0.1827]) 
zonotope_slice_345(z1,mat['info_FRS'][0] [10], slice_value)

Z_obs = convert_obs_to_zonotope(np.array([1,2]),2.5,0.5)
print("obs center and geneartor",Z_obs.x, Z_obs.G)

A = convert_zono_obs_to_constraint(Z_obs,Z_obs)
#info contain info about which generator to slice during online for each zonotope, in order [theta; thetadot; k]

#don't think about try to viz, you are gonna run out of memory. 
#TODO:viz by doing a project operation: get rid of other dims, summing geneartors to dim 0 and 1, where there is a element
# fig = visZ([z1],title="Zonotopes")
# show(fig) # uncomment if you want to see it

# slice_G = G(slice_dim, slice_idx);
# slice_lambda = slice_G\(slice_pt - slice_c);
# %basically to slice a range, you'll get a "slice_lambda_1" for the lower bound of the range, and "slice_lambda_2" for the upper bound
# if size(slice_lambda, 2) > 1
#     error('slice_lambda is not 1D');
# end
# if any(abs(slice_lambda) > 1)
# %      warning(num2str(slice_lambda(:)')+"Slice point is outside bounds of reach set, and therefore is not verified");
#      if any(abs(slice_lambda) > 1.02)
#          error(num2str(slice_lambda(:)')+"Slice point is outside bounds of reach set, and therefore is not verified");
#      end
    
#     slice_lambda(slice_lambda > 1) = 1;
#     slice_lambda(slice_lambda < -1) = -1;
# end

# newG = G;
# newG(:, slice_idx) = [];
# newc = c + G(:, slice_idx)*slice_lambda;

# newzono = zonotope([newc, newG]);
