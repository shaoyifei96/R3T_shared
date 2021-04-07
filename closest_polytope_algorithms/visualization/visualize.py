import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from closest_polytope_algorithms.bounding_box.box import AABB
from matplotlib.collections import PatchCollection
from scipy.spatial import voronoi_plot_2d
from pypolycontain.visualization.visualize_2D import visualize_2D_zonotopes as visZ


def visualize_boxes(box_list, dim_x = 0, dim_y = 1, xlim=None, ylim=None, ax = None, fig = None,alpha=None,linewidth=3, facecolor='black'):
    if ax is None:
        fig, ax = plt.subplots(1)

    for box in box_list:
        if isinstance(box, AABB):
            x = box.u[dim_x]
            y = box.u[dim_y]
            width = box.v[dim_x] - box.u[dim_x]
            height = box.v[dim_y] - box.u[dim_y]
        else:
            # lower corner - upper corner representation
            flattened_box = np.ndarray.flatten(box)
            # print(flattened_box)
            dim = int(flattened_box.shape[0]/2)
            width = abs(flattened_box[dim_x] - flattened_box[dim_x+dim])
            height = abs(flattened_box[dim_y] - flattened_box[dim_y+dim])
            x = (flattened_box[dim_x] + flattened_box[dim_x+dim])/2-width/2
            y = (flattened_box[dim_y] + flattened_box[dim_y+dim])/2-height/2
        # if not fill:
            #FIXME: not working
            # rect = patches.Rectangle((x,y), width, height,linewidth=linewidth, alpha=alpha, facecolor=facecolor)
        rect = patches.Rectangle((x, y), width, height, linewidth=linewidth, facecolor=facecolor,alpha=alpha)
        ax.add_patch(rect)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    return fig,ax

def visualize_box_nodes(box_nodes_list, dim_x = 0, dim_y = 1, xlim=None, ylim=None, ax = None, fig = None,alpha=None,fill=False,linewidth=3):
    box_list = []
    for box_node in box_nodes_list:
        # box_list.append(box_node.box)  #original
        box_list.append(box_node)  #hardik:mod
    return visualize_boxes(box_list,dim_x = dim_x, dim_y = dim_y,linewidth=linewidth, xlim=xlim, ylim=ylim, ax = ax, fig = fig,alpha=alpha)#, fill=fill) #hardik:mod

