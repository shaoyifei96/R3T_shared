import numpy as np
from pypolycontain.lib.zonotope import zonotope, zonotope_directed_distance
from pypolycontain.visualization.visualize_2D import visualize_2D_zonotopes as visZ
from matplotlib.pyplot import show

# for matab import, zonotope array saved as cell arrary of matrix, dim x num_generator +1, MATLAB: c = Z(:,1) G = Z(:,2:end)
import scipy.io


def zonotope_slice_345(z, slice_idx, slice_value):
    slice_dim = np.array([2, 3, 4])
    slice_G = z.G[slice_dim, slice_idx]
    # print(slice_G.shape)
    # print(z.G[:, slice_idx].shape)

    # MATLAB: slice_G\(slice_pt - slice_c);
    slice_lambda = np.linalg.solve(slice_G, slice_value - z.x[slice_dim])
    # print(slice_lambda.shape)
    newG = np.delete(z.G, slice_idx, 1)
    newc = np.matmul(z.G[:, slice_idx].squeeze(), slice_lambda) + z.x
    print(newc)

    return zonotope(newc, newG, color="red")



# def zonotope_slice_34(z,slice_idx,slice_value):
#     slice_dim = np.array([2,3])#slice number -1 since python
#     slice_G = z.G[slice_dim, slice_idx]

#     slice_lambda = np.linalg.solve(slice_G, slice_value - z.x[slice_dim])
#     newc =  np.matmul(z.G[:, slice_idx].squeeze(),slice_lambda) + z.x
#     print(f"slide_idx:{slice_idx}")
#     print(f"New center:{newc}")

#     return zonotope(newc,newG,color="red")


if __name__ == "__main__":

    mat = scipy.io.loadmat(
        "/media/hardik/Windows/Ubuntu/R3T_shared/r3t/overapproximate_with_slice/test_zono.mat"
    )

    # time
    print("time intervals", mat["save_FRS"][0].shape)

    # Zonotope Center
    x_1 = mat["save_FRS"][0][10][:, 0]
    print("center", x_1)

    # generators to slice for the 10th time interval
    G_1 = mat["save_FRS"][0][10][:, 1:]
    print(G_1.shape)

    # info about which generator to slice during online for each zonotope, in order [theta; thetadot; k]
    print(mat["info_FRS"][0][10])

    z1 = zonotope(x_1, G_1, color="green")
    print(z1)
    slice_value = np.array([0.1324, -0.4025, 0.1827])
    zonotope_slice_345(z1, mat["info_FRS"][0][10], slice_value)

    # don't think about try to viz, you are gonna run out of memory.
    # TODO:viz by doing a project operation: get rid of other dims, summing geneartors to dim 0 and 1, where there is a element
    # fig = visZ([z1],title="Zonotopes")
    # show(fig) # uncomment if you want to see it

    # slice_G = G(slice_dim, slice_idx);
    # slice_lambda = slice_G\(slice_pt - slice_c);
    # # to slice a range, you'll get a "slice_lambda_1" for the lower bound of the range, and "slice_lambda_2" for the upper bound
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
