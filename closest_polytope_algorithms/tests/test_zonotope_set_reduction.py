import numpy as np
from bounding_box.zonotope_tree import *
from visualization.visualize import *
from pypolycontain.visualization.visualize_2D import visualize_2D_zonotopes as visZ

def test_with_dataset():
    #FIXME: Import issues
    from utils.PWA_Control_utils import polytree_to_zonotope_tree
    import pickle
    with open("zonotope_datasets/inverted_pendulum.pkl", "rb") as f:
        state_tree = pickle.load(f)
    f.close()
    # extract zonotopes
    zonotope_tree = polytree_to_zonotope_tree(state_tree)
    zonotope_count = len(zonotope_tree.zonotopes)

    query_point = np.asarray([(np.random.rand(1) - 0.5) * 0.24,
                              (np.random.rand(1) - 0.5) * 2])
    print(query_point)
    query_point = query_point.reshape(-1, 1)
    closest_zonotope, candidate_boxes, query_box = zonotope_tree.find_closest_polytopes(query_point)
    print(('Query point: ', query_point))
    ax_lim = np.asarray([-zonotope_count, zonotope_count, -zonotope_count, zonotope_count]) * 1.1
    fig, ax = visZ(zonotope_tree.zonotopes, title="", alpha=0.2, axis_limit=ax_lim)
    fig, ax = visZ(closest_zonotope, title="", fig=fig, ax=ax, alpha=1, axis_limit=ax_lim)
    fig, ax = visualize_box_nodes(zonotope_tree.box_nodes, fig=fig, ax=ax, alpha=0.4, linewidth=0.5)
    fig, ax = visualize_boxes(candidate_boxes, fig=fig, ax=ax, alpha=1)
    print(('Evaluating %d zonotopes out of %d' % (len(candidate_boxes), len(zonotope_tree.zonotopes))))
    fig, ax = visualize_boxes([query_box], fig=fig, ax=ax, alpha=0.3, fill=True)
    plt.scatter(query_point[0], query_point[1], s=20, color='k')
    print(('Closest Zonotope: ', closest_zonotope))
    plt.show()

def zonotope_reduction_line_nd(zonotope_count, dim, num_of_queries):
    zonotopes = []
    centroid_range = zonotope_count*5
    generator_range = 10
    for i in range(zonotope_count):
        m = np.random.random_integers(dim, 2*dim)
        G = 2*(np.random.rand(dim, m) - 0.5) * generator_range * 1
        x = 2*(np.random.rand(dim-1,1) - 0.5)
        x = np.vstack([2*(np.random.rand(1,1) - 0.5) * centroid_range,x])
        # print(x)
        zonotopes.append(zonotope(x, G))
    zt = PolytopeTree_Old(zonotopes)

    query_points = (np.random.rand(dim-1,num_of_queries)-0.5)
    query_points = np.vstack([2*(np.random.rand(1, num_of_queries) - 0.5) * centroid_range,query_points])
    # print(query_points)
    reduction_ratios = np.zeros([num_of_queries])
    for i, q in enumerate(query_points.T):
        closest_zonotope, candidate_boxes, query_box = zt.find_closest_polytopes(q)
        reduction_ratios[i] = len(candidate_boxes)/(zonotope_count*1.)

    if dim ==2:
        fig, ax = visZ(zt.polytopes, title="", alpha=0.2)
        ax.scatter(query_points[0,:], query_points[1,:],s=3)
        ax.set_xlim([-centroid_range*1.2,centroid_range*1.2])
        ax.set_ylim([-centroid_range * 1.2, centroid_range * 1.2])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Zonotope and Query Point Distribution')

        plt.show()

    return np.average(reduction_ratios), np.std(reduction_ratios)

def zonotope_reduction_line_over_d(zonotope_count, d_start,d_end, num_of_queries):
    d = np.arange(d_start,d_end)
    reduction_ratios = np.zeros([d.shape[0],2])
    for i, d_i in enumerate(d):
        print(('Evaluating %d-D' %d_i))
        reduction_ratios[i,0],reduction_ratios[i,1] = zonotope_reduction_line_nd(d_i**zonotope_count,d[i],num_of_queries)
    print(reduction_ratios)
    plt.errorbar(d,100*reduction_ratios[:,0],100*reduction_ratios[:,1],marker='.',ecolor='r',elinewidth=0.3,capsize=2,linewidth=0.5,markersize=7)
    plt.xlabel('Dimension')
    plt.ylabel('% of zonotopes evaluated')
    plt.title('Line Distributed Zonotope Set Reduction with Bounding Boxes')
    plt.show()

def zonotope_reduction_box_nd(zonotope_count, dim, num_of_queries):
    zonotopes = []
    centroid_range = zonotope_count*8
    generator_range = zonotope_count
    print(('Centroid range: %d' %centroid_range))
    print(('Generator range: %d' %generator_range))
    for i in range(zonotope_count):
        m = np.random.random_integers(dim, 2*dim)
        G = (np.random.rand(dim, m) - 0.5) * generator_range * 1
        x = 2*(np.random.rand(dim,1) - 0.5)*centroid_range
        # print(x)
        zonotopes.append(zonotope(x, G))
    zt = PolytopeTree_Old(zonotopes)
    #query points in a line
    # query_points = (np.random.rand(dim-1,num_of_queries)-0.5)
    # query_points = np.vstack([query_points, 2*(np.random.rand(1, num_of_queries) - 0.5) * centroid_range*2])
    #query points anywhere in space
    query_points = 2 * (np.random.rand(dim, num_of_queries) - 0.5) * centroid_range * 2
    reduction_ratios = np.zeros([num_of_queries])
    for i, q in enumerate(query_points.T):
        closest_zonotope, candidate_boxes, query_box = zt.find_closest_polytopes(q)
        reduction_ratios[i] = len(candidate_boxes)/(zonotope_count*1.)
    if dim ==2:
        fig, ax = visZ(zt.polytopes, title="", alpha=0.2)
        ax.scatter(query_points[0,:], query_points[1,:],s=3)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Zonotope and Query Point Distribution')

        plt.show()
    return np.average(reduction_ratios), np.std(reduction_ratios)

def zonotope_reduction_box_over_d(zonotope_count, d_start,d_end, num_of_queries):
    d = np.arange(d_start,d_end)
    reduction_ratios = np.zeros([d.shape[0],2])
    for i, d_i in enumerate(d):
        print(('Evaluating %d-D' %d_i))
        reduction_ratios[i,0],reduction_ratios[i,1] = zonotope_reduction_box_nd(d_i**zonotope_count,d[i],num_of_queries)
    print(reduction_ratios)
    plt.errorbar(d,100*reduction_ratios[:,0],100*reduction_ratios[:,1],marker='.',ecolor='r',elinewidth=0.3,capsize=2,linewidth=0.5,markersize=7)
    plt.xlabel('Dimension')
    plt.ylabel('% of zonotopes evaluated')
    plt.title('Box Distributed Zonotope Set Reduction with Bounding Boxes')
    plt.show()


if __name__ == '__main__':
    plt.rcParams["font.family"] = "Times New Roman"
    zonotope_reduction_box_over_d(5,2,5,100)

