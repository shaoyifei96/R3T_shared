import random
import timeit

from bounding_box.zonotope_tree import *
from visualization.visualize import *

space_size = 10000
def time_construct_zonotope_tree(zonotope_count, dim):
    zonotopes = []
    centroid_range = zonotope_count
    generator_range = 3
    for i in range(zonotope_count):
        m = np.random.random_integers(dim, 3*dim)
        G = (np.random.rand(dim, m) - 0.5) * generator_range
        x = (np.random.rand(dim, 1) - 0.5) * centroid_range
        zonotopes.append(zonotope(x, G))
    start_time = timeit.default_timer()
    zt = PolytopeTree_Old(zonotopes)
    end_time = timeit.default_timer()

    return end_time-start_time, zt

def time_query_zonotope_tree(zonotope_count, dim, num_of_queries):
    zonotopes = []
    centroid_range = zonotope_count
    generator_range = 3
    for i in range(zonotope_count):
        m = np.random.random_integers(dim, 3*dim)
        G = (np.random.rand(dim, m) - 0.5) * generator_range
        x = (np.random.rand(dim, 1) - 0.5) * centroid_range
        zonotopes.append(zonotope(x, G))
    zt = PolytopeTree_Old(zonotopes)
    query_points = (np.random.rand(num_of_queries,dim)-0.5)*centroid_range
    start_time = timeit.default_timer()
    for q in query_points:
        zt.find_closest_polytopes(q)
    end_time = timeit.default_timer()
    return end_time-start_time

def time_query_zonotope_tree_line(zonotope_count, dim, num_of_queries):
    zonotopes = []
    centroid_range = zonotope_count*4
    generator_range = 3
    for i in range(zonotope_count):
        m = np.random.random_integers(dim, 2*dim)
        G = (np.random.rand(dim, m) - 0.5) * generator_range * 1
        x = np.random.rand(dim-1,1) - 0.5
        x = np.vstack([x, (np.random.rand(1,1) - 0.5) * centroid_range])
        # print(x)
        zonotopes.append(zonotope(x, G))
    zt = PolytopeTree_Old(zonotopes)

    query_points = (np.random.rand(dim-1,num_of_queries)-0.5)
    query_points = np.vstack([query_points, (np.random.rand(1, num_of_queries) - 0.5) * centroid_range])
    # print(query_points)
    start_time = timeit.default_timer()
    for q in query_points.T:
        zt.find_closest_polytopes(q)
    end_time = time.time()
    return end_time-start_time

def query_time_over_zonotope_count():
    #parameters
    repeats = 2
    dimensions = 10
    start_zonotope_count = 1
    end_zonotope_count = 3
    num = 3
    queries_count = 100

    #time constructions
    zonotope_count = np.rint(np.logspace(start_zonotope_count,end_zonotope_count,num))
    runtime = np.zeros((num,repeats))
    runtime_avg = np.zeros((num, 1))
    runtime_stdev = np.zeros((num,1))

    for i, bc in enumerate(zonotope_count):
        print(('Starting bounding_box count ', bc))
        for r in range(repeats):
            print(('Repeat ', r))
            runtime[i, r] = time_query_zonotope_tree_line(int(bc),dimensions,queries_count)
            print(('Ran for ', runtime[i, r]))
        runtime_avg = np.average(runtime, axis=1)
    plt.subplot(2,1,1)
    plt.semilogx(zonotope_count, runtime_avg)
    plt.xlabel('$log$ Number of zonotopes')
    plt.ylabel('Runtime(s)')
    plt.title('Query time of %d points vs $log$ zonotope count in $R^{%d}$' % (queries_count,dimensions))
    plt.subplot(2,1,2)
    plt.loglog(zonotope_count, runtime_avg)
    plt.xlabel('$log$ Number of zonotopes')
    plt.ylabel('$log$ Runtime(s)')
    plt.title('$log$ Query time of %d points vs $log$ zonotope count in $R^{%d}$' % (queries_count,dimensions))
    plt.show()
#
# def box_construction_over_runtime_dimensions():
#     #time constructions
#     #parameters
#     box_counts = 10000
#     dimensions = np.arange(2,50)
#     num = len(dimensions)
#     repeats = 3
#
#     runtime = np.zeros((num,repeats))
#     runtime_avg = np.zeros((num, 1))
#     runtime_stdev = np.zeros((num,1))
#
#     for i, dim in enumerate(dimensions):
#         print('Starting dimensions ', dim)
#         for r in range(repeats):
#             print('Repeat ', r)
#             runtime[i, r],_root,_bnl = time_construct_box_tree(box_counts, dim)
#             print('Ran for ', runtime[i, r])
#         runtime_avg = np.average(runtime, axis=1)
#     plt.subplot(2,1,1)
#     plt.plot(dimensions, runtime_avg)
#     plt.xlabel('Dimension')
#     plt.ylabel('Runtime(s)')
#     plt.title('Runtime vs dimension with $10^{%d}$ boxes' % np.log10(box_counts))
#     plt.subplot(2,1,2)
#     plt.semilogy(dimensions, runtime_avg)
#     plt.xlabel('Dimension')
#     plt.ylabel('$log$ Runtime(s)')
#     plt.title('$log$ Runtime vs dimension with %d boxes' % box_counts)
#     plt.show()
#
# def box_query_over_box_count():
#     #parameters
#     repeats = 3
#     dimensions = 5
#     start_box_count = 1
#     end_box_count = 5
#     num = 10
#     queries = 1000
#
#     #time constructions
#     box_counts = np.rint(np.logspace(start_box_count,end_box_count,num))
#     runtime = np.zeros((num,repeats))
#     runtime_avg = np.zeros((num, 1))
#     runtime_stdev = np.zeros((num,1))
#
#     for i, bc in enumerate(box_counts):
#         print('Starting bounding_box count ', bc)
#         for r in range(repeats):
#             print('Repeat ', r)
#             _rt,root,bnl = time_construct_box_tree(int(bc),dimensions)
#             runtime[i, r] = test_closest_box(root,queries)
#             print('Ran for ', runtime[i, r])
#         runtime_avg = np.average(runtime, axis=1)
#     plt.subplot(2,1,1)
#     plt.semilogx(box_counts, runtime_avg)
#     plt.xlabel('$log$ Number of boxes')
#     plt.ylabel('Runtime(s)')
#     plt.title('Runtime for %d queries vs $log$ bounding_box count in $R^{%d}$' % (queries, dimensions))
#     plt.subplot(2,1,2)
#     plt.loglog(box_counts, runtime_avg)
#     plt.xlabel('$log$ Number of boxes')
#     plt.ylabel('$log$ Runtime(s)')
#     plt.title('$log$ Runtime for %d queries vs $log$ queries in $R^{%d}$' % (queries, dimensions))
#     plt.show()
#
# def box_query_over_runtime_dimensions():
#     #time constructions
#     #parameters
#     box_counts = 10000
#     dimensions = np.arange(2,50)
#     num = len(dimensions)
#     repeats = 3
#     queries = 1000
#
#     runtime = np.zeros((num,repeats))
#     runtime_avg = np.zeros((num, 1))
#     runtime_stdev = np.zeros((num,1))
#
#     for i, dim in enumerate(dimensions):
#         print('Starting dimensions ', dim)
#         for r in range(repeats):
#             print('Repeat ', r)
#             _rt,root,bnl = time_construct_box_tree(box_counts,dim)
#             runtime[i, r] = test_closest_box(root,queries)
#             print('Ran for ', runtime[i, r])
#         runtime_avg = np.average(runtime, axis=1)
#     plt.subplot(2,1,1)
#     plt.plot(dimensions, runtime_avg)
#     plt.xlabel('Dimension')
#     plt.ylabel('Runtime(s)')
#     plt.title('Runtime for %d queries vs dimension with %d boxes' % (queries, box_counts))
#     plt.subplot(2,1,2)
#     plt.semilogy(dimensions, runtime_avg)
#     plt.xlabel('Dimension')
#     plt.ylabel('$log$ Runtime(s)')
#     plt.title('$log$ Runtime for %d queries vs dimension with %d boxes' % (queries, box_counts))
#     plt.show()
#

if __name__ == '__main__':
    # runtime_box_count()
    query_time_over_zonotope_count()