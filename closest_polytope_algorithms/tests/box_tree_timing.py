import random
import timeit

from bounding_box.box import *
from bounding_box.box_tree import *
from visualization.visualize import *

space_size = 10000
def time_construct_box_tree(box_count, dim):
    box_list = []
    box_node_list = []
    for i in range(box_count):
        xs = random.sample(list(range(space_size)), dim)
        ys = random.sample(list(range(space_size)),dim)
        uv = np.zeros((dim,2))
        for j in range(dim):
            uv[j,0] = xs[j]
            # uv[j,1] = ys[j]
            uv[j,1] = xs[j]+abs(np.random.normal(scale=5)+10) #FIXME: Random bounding_box generation

        box = AABB([uv[:,0],uv[:,1]])
        # print(bounding_box)
        box_list.append(box)
        box_node_list.append(BoxNode(box))
    start_time = timeit.default_timer()
    root = binary_split(box_node_list)
    end_time = timeit.default_timer()
    return end_time-start_time, root, box_node_list

def test_closest_box(root, num_of_queries):
    query_boxes = []
    dim = root.box.u.shape[0]
    for i in range(num_of_queries):
        xs = random.sample(list(range(space_size)), dim)
        ys = random.sample(list(range(space_size)),dim)
        uv = np.zeros((dim,2))
        for j in range(dim):
            uv[j,0] = xs[j]
            # uv[j,1] = ys[j]
            uv[j,1] = xs[j]+abs(np.random.normal(scale=5)+10) #FIXME: Random bounding_box generation

        box = AABB([uv[:,0],uv[:,1]])
        query_boxes.append(box)
    start_time = timeit.default_timer()
    for box in query_boxes:
        l = []
        root.evaluate_node(box,l)
    end_time = timeit.default_timer()
    return end_time-start_time

def box_construction_over_box_count():
    #parameters
    repeats = 3
    dimensions = 5
    start_box_count = 1
    end_box_count = 5
    num = 10

    #time constructions
    box_counts = np.rint(np.logspace(start_box_count,end_box_count,num))
    runtime = np.zeros((num,repeats))
    runtime_avg = np.zeros((num, 1))
    runtime_stdev = np.zeros((num,1))

    for i, bc in enumerate(box_counts):
        print(('Starting bounding_box count ', bc))
        for r in range(repeats):
            print(('Repeat ', r))
            runtime[i, r],_root,_bnl = time_construct_box_tree(int(bc),dimensions)
            print(('Ran for ', runtime[i, r]))
        runtime_avg = np.average(runtime, axis=1)
    plt.subplot(2,1,1)
    plt.semilogx(box_counts, runtime_avg)
    plt.xlabel('$log$ Number of boxes')
    plt.ylabel('Runtime(s)')
    plt.title('Runtime vs $log$ bounding_box count in $R^{%d}$' % dimensions)
    plt.subplot(2,1,2)
    plt.loglog(box_counts, runtime_avg)
    plt.xlabel('$log$ Number of boxes')
    plt.ylabel('$log$ Runtime(s)')
    plt.title('$log$ Runtime vs $log$ bounding_box count in $R^{%d}$' % dimensions)
    plt.show()

def box_construction_over_runtime_dimensions():
    #time constructions
    #parameters
    box_counts = 10000
    dimensions = np.arange(2,50)
    num = len(dimensions)
    repeats = 3

    runtime = np.zeros((num,repeats))
    runtime_avg = np.zeros((num, 1))
    runtime_stdev = np.zeros((num,1))

    for i, dim in enumerate(dimensions):
        print(('Starting dimensions ', dim))
        for r in range(repeats):
            print(('Repeat ', r))
            runtime[i, r],_root,_bnl = time_construct_box_tree(box_counts, dim)
            print(('Ran for ', runtime[i, r]))
        runtime_avg = np.average(runtime, axis=1)
    plt.subplot(2,1,1)
    plt.plot(dimensions, runtime_avg)
    plt.xlabel('Dimension')
    plt.ylabel('Runtime(s)')
    plt.title('Runtime vs dimension with $10^{%d}$ boxes' % np.log10(box_counts))
    plt.subplot(2,1,2)
    plt.semilogy(dimensions, runtime_avg)
    plt.xlabel('Dimension')
    plt.ylabel('$log$ Runtime(s)')
    plt.title('$log$ Runtime vs dimension with %d boxes' % box_counts)
    plt.show()

def box_query_over_box_count():
    #parameters
    repeats = 3
    dimensions = 5
    start_box_count = 1
    end_box_count = 5
    num = 10
    queries = 1000

    #time constructions
    box_counts = np.rint(np.logspace(start_box_count,end_box_count,num))
    runtime = np.zeros((num,repeats))
    runtime_avg = np.zeros((num, 1))
    runtime_stdev = np.zeros((num,1))

    for i, bc in enumerate(box_counts):
        print(('Starting bounding_box count ', bc))
        for r in range(repeats):
            print(('Repeat ', r))
            _rt,root,bnl = time_construct_box_tree(int(bc),dimensions)
            runtime[i, r] = test_closest_box(root,queries)
            print(('Ran for ', runtime[i, r]))
        runtime_avg = np.average(runtime, axis=1)
    plt.subplot(2,1,1)
    plt.semilogx(box_counts, runtime_avg)
    plt.xlabel('$log$ Number of boxes')
    plt.ylabel('Runtime(s)')
    plt.title('Runtime for %d queries vs $log$ bounding_box count in $R^{%d}$' % (queries, dimensions))
    plt.subplot(2,1,2)
    plt.loglog(box_counts, runtime_avg)
    plt.xlabel('$log$ Number of boxes')
    plt.ylabel('$log$ Runtime(s)')
    plt.title('$log$ Runtime for %d queries vs $log$ queries in $R^{%d}$' % (queries, dimensions))
    plt.show()

def box_query_over_runtime_dimensions():
    #time constructions
    #parameters
    box_counts = 10000
    dimensions = np.arange(2,50)
    num = len(dimensions)
    repeats = 3
    queries = 1000

    runtime = np.zeros((num,repeats))
    runtime_avg = np.zeros((num, 1))
    runtime_stdev = np.zeros((num,1))

    for i, dim in enumerate(dimensions):
        print(('Starting dimensions ', dim))
        for r in range(repeats):
            print(('Repeat ', r))
            _rt,root,bnl = time_construct_box_tree(box_counts,dim)
            runtime[i, r] = test_closest_box(root,queries)
            print(('Ran for ', runtime[i, r]))
        runtime_avg = np.average(runtime, axis=1)
    plt.subplot(2,1,1)
    plt.plot(dimensions, runtime_avg)
    plt.xlabel('Dimension')
    plt.ylabel('Runtime(s)')
    plt.title('Runtime for %d queries vs dimension with %d boxes' % (queries, box_counts))
    plt.subplot(2,1,2)
    plt.semilogy(dimensions, runtime_avg)
    plt.xlabel('Dimension')
    plt.ylabel('$log$ Runtime(s)')
    plt.title('$log$ Runtime for %d queries vs dimension with %d boxes' % (queries, box_counts))
    plt.show()


if __name__ == '__main__':
    # runtime_box_count()
    box_query_over_runtime_dimensions()