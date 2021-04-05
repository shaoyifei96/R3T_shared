from closest_polytope_algorithms.voronoi.voronoi import *
from pypolycontain.utils.random_polytope_generator import *
from closest_polytope_algorithms.bounding_box.polytope_tree import PolytopeTree
from pypolycontain.visualization.visualize_2D import visualize_2D_zonotopes as visZ
from closest_polytope_algorithms.utils.polytope_dataset_utils import *
from scipy.spatial import voronoi_plot_2d
from timeit import default_timer
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import os
import time
from datetime import datetime

def test_random_zonotope_count(dim=2, counts = np.arange(3, 16, 3)*10, construction_repeats = 1, queries=100,save=False, random_zonotope_generator = get_uniform_random_zonotopes):

    voronoi_precomputation_times = np.zeros([len(counts), construction_repeats])
    voronoi_query_times = np.zeros([len(counts), construction_repeats*queries])
    voronoi_query_reduction_percentages = np.zeros([len(counts), construction_repeats*queries])

    aabb_precomputation_times = np.zeros([len(counts), construction_repeats])
    aabb_query_times = np.zeros([len(counts), construction_repeats*queries])
    aabb_query_reduction_percentages = np.zeros([len(counts), construction_repeats*queries])
    line_width = 10
    seed=int(time.time())
    for cr_index in range(construction_repeats):
        print(('Repetition %d' %cr_index))
        for count_index, count in enumerate(counts):
            print(('Testing %d zonotopes...' % count))

            if random_zonotope_generator==get_uniform_random_zonotopes:
                zonotopes = random_zonotope_generator(count, dim=dim, generator_range=count * 0.3,
                                                         centroid_range=count * 1.5, return_type='zonotope', seed=seed)
            # # line params
            elif random_zonotope_generator==get_line_random_zonotopes:
                zonotopes = random_zonotope_generator(count, dim=dim, line_width = line_width, generator_range=count * 0.3,
                                                         centroid_range=count * 1.5, return_type='zonotope', seed=seed)
            else:
                raise NotImplementedError
            #test voronoi
            construction_start_time = default_timer()
            vcp = VoronoiClosestPolytope(zonotopes)
            voronoi_precomputation_times[count_index, cr_index] = default_timer()-construction_start_time
            #query
            for query_index in range(queries):
                if random_zonotope_generator==get_uniform_random_zonotopes:
                    query_point = (np.random.rand(dim) - 0.5) * count * 5 #random query point
                elif random_zonotope_generator==get_line_random_zonotopes:
                    query_point = (np.random.rand(dim) - 0.5)
                    query_point[1:] = query_point[1:] * 2 * line_width * 3
                    query_point[0] = query_point[0] * 2 * count * 2  # random query point
                else:
                    raise NotImplementedError
                query_start_time = default_timer()
                best_zonotope, best_distance, evaluated_zonotopes = vcp.find_closest_polytope(query_point, return_intermediate_info=True)
                voronoi_query_times[count_index,cr_index*queries+query_index] = default_timer()-query_start_time
                voronoi_query_reduction_percentages[count_index, cr_index*queries+query_index] = len(evaluated_zonotopes)*100./count

            #test aabb
            construction_start_time = default_timer()
            zono_tree = PolytopeTree(zonotopes)
            aabb_precomputation_times[count_index, cr_index] = default_timer()-construction_start_time
            #query
            for query_index in range(queries):
                if random_zonotope_generator==get_uniform_random_zonotopes:
                    query_point = (np.random.rand(dim) - 0.5) * count * 5 #random query point
                elif random_zonotope_generator==get_line_random_zonotopes:
                    query_point = (np.random.rand(dim) - 0.5)
                    query_point[1:] = query_point[1:] * 2 * line_width * 3
                    query_point[0] = query_point[0] * 2 * count * 2  # random query point
                else:
                    raise NotImplementedError
                query_start_time = default_timer()
                best_zonotope, best_distance, evaluated_zonotopes, query_box = zono_tree.find_closest_polytopes(query_point, return_intermediate_info=True)
                aabb_query_times[count_index,cr_index*queries+query_index] = default_timer()-query_start_time
                aabb_query_reduction_percentages[count_index, cr_index*queries+query_index] = len(evaluated_zonotopes)*100./count


    voronoi_precomputation_times_median = np.median(voronoi_precomputation_times, axis=1)
    voronoi_precomputation_times_min = np.min(voronoi_precomputation_times, axis=1)
    voronoi_precomputation_times_max = np.max(voronoi_precomputation_times, axis=1)

    voronoi_query_times_median = np.median(voronoi_query_times, axis=1)
    voronoi_query_times_min = np.min(voronoi_query_times, axis=1)
    voronoi_query_times_max = np.max(voronoi_query_times, axis=1)

    voronoi_query_reduction_percentages_median = np.median(voronoi_query_reduction_percentages, axis=1)
    voronoi_query_reduction_percentages_min = np.min(voronoi_query_reduction_percentages, axis=1)
    voronoi_query_reduction_percentages_max = np.max(voronoi_query_reduction_percentages, axis=1)

    aabb_precomputation_times_median = np.median(aabb_precomputation_times, axis=1)
    aabb_precomputation_times_min = np.min(aabb_precomputation_times, axis=1)
    aabb_precomputation_times_max = np.max(aabb_precomputation_times, axis=1)

    aabb_query_times_median = np.median(aabb_query_times, axis=1)
    aabb_query_times_min = np.min(aabb_query_times, axis=1)
    aabb_query_times_max = np.max(aabb_query_times, axis=1)

    aabb_query_reduction_percentages_median = np.median(aabb_query_reduction_percentages, axis=1)
    aabb_query_reduction_percentages_min = np.min(aabb_query_reduction_percentages, axis=1)
    aabb_query_reduction_percentages_max = np.max(aabb_query_reduction_percentages, axis=1)

    #save data
    experiment_name = datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H-%M-%S')
    os.makedirs('test_random_zonotope_count'+experiment_name)
    np.save('test_random_zonotope_count'+experiment_name+'/voronoi_precomputation_times_median', voronoi_precomputation_times_median)
    np.save('test_random_zonotope_count'+experiment_name+'/voronoi_precomputation_times_min', voronoi_precomputation_times_min)
    np.save('test_random_zonotope_count'+experiment_name+'/voronoi_precomputation_times_max', voronoi_precomputation_times_max)

    np.save('test_random_zonotope_count'+experiment_name+'/voronoi_query_times_median', voronoi_query_times_median)
    np.save('test_random_zonotope_count'+experiment_name+'/voronoi_query_times_min', voronoi_query_times_min)
    np.save('test_random_zonotope_count'+experiment_name+'/voronoi_query_times_max', voronoi_query_times_max)
    np.save('test_random_zonotope_count'+experiment_name+'/voronoi_query_reduction_percentages_median', voronoi_query_reduction_percentages_median)
    np.save('test_random_zonotope_count'+experiment_name+'/voronoi_query_reduction_percentages_min', voronoi_query_reduction_percentages_min)
    np.save('test_random_zonotope_count'+experiment_name+'/voronoi_query_reduction_percentages_max', voronoi_query_reduction_percentages_max)

    np.save('test_random_zonotope_count'+experiment_name+'/aabb_precomputation_times_median', aabb_precomputation_times_median)
    np.save('test_random_zonotope_count'+experiment_name+'/aabb_precomputation_times_min', aabb_precomputation_times_min)
    np.save('test_random_zonotope_count'+experiment_name+'/aabb_precomputation_times_max', aabb_precomputation_times_max)

    np.save('test_random_zonotope_count'+experiment_name+'/aabb_query_times_median', aabb_query_times_median)
    np.save('test_random_zonotope_count'+experiment_name+'/aabb_query_times_min', aabb_query_times_min)
    np.save('test_random_zonotope_count'+experiment_name+'/aabb_query_times_max', aabb_query_times_max)

    np.save('test_random_zonotope_count'+experiment_name+'/aabb_query_reduction_percentages_median', aabb_query_reduction_percentages_median)
    np.save('test_random_zonotope_count'+experiment_name+'/aabb_query_reduction_percentages_min', aabb_query_reduction_percentages_min)
    np.save('test_random_zonotope_count'+experiment_name+'/aabb_query_reduction_percentages_max', aabb_query_reduction_percentages_max)

    params = np.array([['dim', np.atleast_1d(dim)],['count', np.atleast_1d(counts)], ['construction_repeats', np.atleast_1d(construction_repeats)], \
                       ['queries', np.atleast_1d(queries)],['seed',np.atleast_1d(seed)], ['random_zonotope_generator', random_zonotope_generator.__name__]])
    np.save('test_random_zonotope_count'+experiment_name+'/params', params)


    #plots
    fig_index = 0
    plt.figure(fig_index)
    fig_index+=1
    plt.subplot(111)
    plt.errorbar(counts, voronoi_precomputation_times_median, np.vstack([voronoi_precomputation_times_median-voronoi_precomputation_times_min,
                                                                       voronoi_precomputation_times_max-voronoi_precomputation_times_median]), marker='.', color = 'b', ecolor='b', elinewidth=0.3,
                 capsize=2, linewidth=0.5, markersize=7)
    plt.errorbar(counts, aabb_precomputation_times_median, np.vstack([aabb_precomputation_times_median-aabb_precomputation_times_min,
                                                                    aabb_precomputation_times_max-aabb_precomputation_times_median]), marker='.', color='r', ecolor='r', elinewidth=0.3,
                 capsize=2, linewidth=0.5, markersize=7)
    plt.legend(['Triangle Ineq.', 'AABB'])

    plt.xlabel('Zonotope Count')
    plt.ylabel('Precomputation Time (s)')
    plt.title('Closest Zonotope Precomputation Time in %d-D' %dim)
    #
    # plt.subplot(212)
    # plt.plot(np.log(counts),np.log(voronoi_precomputation_times_avg))
    # plt.xlabel('$log$ Zonotope Count')
    # plt.ylabel('$log$ Precomputation Time (s)')
    # plt.title('$log$ Voronoi Closest Zonotope Precomputation Time in %d-D' %dim)
    # plt.tight_layout()
    if save:
        plt.savefig('precomputation_time' + str(default_timer()) + '.png', dpi=500)

    plt.figure(fig_index)
    fig_index+=1
    plt.subplot(111)
    plt.errorbar(counts, voronoi_query_times_median, np.vstack([voronoi_query_times_median-voronoi_query_times_min,
                                                              voronoi_query_times_max-voronoi_query_times_median]), marker='.', color='b', ecolor='b', elinewidth=0.3, capsize=2,
                 linewidth=0.5, markersize=7)
    plt.errorbar(counts, aabb_query_times_median, np.vstack([aabb_query_times_median-aabb_query_times_min,
                                                           aabb_query_times_max-aabb_query_times_min]), marker='.', color='r', ecolor='r', elinewidth=0.3, capsize=2,
                 linewidth=0.5, markersize=7)
    plt.legend(['Triangle Ineq.', 'AABB'])
    plt.xlabel('Polytope Count')
    plt.ylabel('Query Time (s)')
    plt.title('Closest Zonotope Single Query Time in %d-D' %dim)
    #
    # plt.subplot(212)
    # plt.plot(np.log(counts),np.log(aabb_query_times_avg))
    # plt.xlabel('$log$ Zonotope Count')
    # plt.ylabel('$log$ Query Time (s)')
    # plt.title('$log$ Voronoi Closest Zonotope Single Query Time in %d-D' %dim)
    # plt.tight_layout()
    if save:
        plt.savefig('query_time' + str(default_timer()) + '.png', dpi=500)

    plt.figure(fig_index)
    fig_index+=1
    plt.subplot(111)
    plt.errorbar(counts, voronoi_query_reduction_percentages_median, np.vstack([voronoi_query_reduction_percentages_median-voronoi_query_reduction_percentages_min,
                                                                              voronoi_query_reduction_percentages_max - voronoi_query_reduction_percentages_median]), marker='.', color='b',ecolor='b',
                 elinewidth=0.3, capsize=2, linewidth=0.5, markersize=7)
    plt.errorbar(counts, aabb_query_reduction_percentages_median, np.vstack([aabb_query_reduction_percentages_median-aabb_query_reduction_percentages_min,
                                                                           aabb_query_reduction_percentages_max-aabb_query_reduction_percentages_median]), marker='.', color='r', ecolor='r',
                 elinewidth=0.3, capsize=2, linewidth=0.5, markersize=7)
    plt.legend(['Triangle Ineq.', 'AABB'])

    plt.xlabel('Polytope Count')
    plt.ylabel('% of Zonotopes Evaluated')
    plt.title('Closest Zonotope Reduction Percentage in %d-D' %dim)
    # plt.ylim(ymin=0)
    #
    # plt.subplot(212)
    # plt.plot(np.log(counts),np.log(voronoi_query_reduction_percentages_avg))
    # plt.xlabel('$log$ Zonotope Count')
    # plt.ylabel('$log$ % of Zonotopes Evaluated')
    # plt.title('$log$ Voronoi Closest Zonotope Reduction Percentage in %d-D' %dim)
    # plt.tight_layout()
    if save:
        plt.savefig('reduction_percentage' + str(default_timer()) + '.png', dpi=500)

    else:
        plt.show()



def test_random_zonotope_dim(count=100, dims=np.arange(2, 11, 1), construction_repeats=1, queries=100, save=False, random_zonotope_generator = get_uniform_random_zonotopes):
    voronoi_precomputation_times = np.zeros([len(dims), construction_repeats])
    voronoi_query_times = np.zeros([len(dims), construction_repeats * queries])
    voronoi_query_reduction_percentages = np.zeros([len(dims), construction_repeats * queries])
    aabb_precomputation_times = np.zeros([len(dims), construction_repeats])
    aabb_query_times = np.zeros([len(dims), construction_repeats * queries])
    aabb_query_reduction_percentages = np.zeros([len(dims), construction_repeats * queries])
    seed = int(time.time())
    # For line distribution
    line_width = 10
    for cr_index in range(construction_repeats):
        print(('Repetition %d' % cr_index))
        for dim_index, dim in enumerate(dims):
            print(('Testing zonotopes in %d-D...' % dim))
            # generate random zonotopes
            # uniform params
            if random_zonotope_generator==get_uniform_random_zonotopes:
                zonotopes = random_zonotope_generator(count, dim=dim, generator_range=count * 0.3,
                                                         centroid_range=count * 1.5, return_type='zonotope', seed=seed)
            # # line params
            elif random_zonotope_generator==get_line_random_zonotopes:
                zonotopes = random_zonotope_generator(count, dim=dim, line_width = line_width, generator_range=count * 0.3,
                                                         centroid_range=count * 1.5, return_type='zonotope', seed=seed)
            else:
                raise NotImplementedError
            # test voronoi
            construction_start_time = default_timer()
            vcp = VoronoiClosestPolytope(zonotopes, max_number_key_points=1000000/count)
            voronoi_precomputation_times[dim_index, cr_index] = default_timer() - construction_start_time
            # query
            for query_index in range(queries):
                if random_zonotope_generator == get_uniform_random_zonotopes:
                    query_point = (np.random.rand(dim) - 0.5) * 2 * count * 2
                elif random_zonotope_generator == get_line_random_zonotopes:
                    query_point = (np.random.rand(dim) - 0.5)
                    query_point[1:] = query_point[1:] * 2 * line_width * 3
                    query_point[0] = query_point[0] * 2 * count * 2  # random query point
                else:
                    raise NotImplementedError
                query_start_time = default_timer()
                best_zonotope, best_distance, evaluated_zonotopes = vcp.find_closest_polytope(query_point,
                                                                                              return_intermediate_info=True)
                voronoi_query_times[dim_index, cr_index * queries + query_index] = default_timer() - query_start_time
                voronoi_query_reduction_percentages[dim_index, cr_index * queries + query_index] = len(
                    evaluated_zonotopes) * 100. / count

            #test aabb
            construction_start_time = default_timer()
            zono_tree = PolytopeTree(zonotopes)
            aabb_precomputation_times[dim_index, cr_index] = default_timer() - construction_start_time
            # query
            for query_index in range(queries):
                if random_zonotope_generator==get_uniform_random_zonotopes:
                    query_point = (np.random.rand(dim) - 0.5)* 2 * count * 2
                elif random_zonotope_generator == get_line_random_zonotopes:
                    query_point = (np.random.rand(dim) - 0.5)
                    query_point[1:] = query_point[1:] * 2 * line_width * 3
                    query_point[0] = query_point[0] * 2 * count * 2  # random query point
                else:
                    raise NotImplementedError
                # print(query_point)
                query_start_time = default_timer()
                best_zonotope, best_distance, evaluated_zonotopes, query_box = zono_tree.find_closest_polytopes(query_point, return_intermediate_info=True)
                # print(len(evaluated_zonotopes))
                aabb_query_times[dim_index, cr_index * queries + query_index] = default_timer() - query_start_time
                aabb_query_reduction_percentages[dim_index, cr_index * queries + query_index] = len(evaluated_zonotopes) * 100. / count

    voronoi_precomputation_times_median = np.median(voronoi_precomputation_times, axis=1)
    voronoi_precomputation_times_min = np.min(voronoi_precomputation_times, axis=1)
    voronoi_precomputation_times_max = np.max(voronoi_precomputation_times, axis=1)

    voronoi_query_times_median = np.median(voronoi_query_times, axis=1)
    voronoi_query_times_min = np.min(voronoi_query_times, axis=1)
    voronoi_query_times_max = np.max(voronoi_query_times, axis=1)

    voronoi_query_reduction_percentages_median = np.median(voronoi_query_reduction_percentages, axis=1)
    voronoi_query_reduction_percentages_min = np.min(voronoi_query_reduction_percentages, axis=1)
    voronoi_query_reduction_percentages_max = np.max(voronoi_query_reduction_percentages, axis=1)

    aabb_precomputation_times_median = np.median(aabb_precomputation_times, axis=1)
    aabb_precomputation_times_min = np.min(aabb_precomputation_times, axis=1)
    aabb_precomputation_times_max = np.max(aabb_precomputation_times, axis=1)

    aabb_query_times_median = np.median(aabb_query_times, axis=1)
    aabb_query_times_min = np.min(aabb_query_times, axis=1)
    aabb_query_times_max = np.max(aabb_query_times, axis=1)

    aabb_query_reduction_percentages_median = np.median(aabb_query_reduction_percentages, axis=1)
    aabb_query_reduction_percentages_min = np.min(aabb_query_reduction_percentages, axis=1)
    aabb_query_reduction_percentages_max = np.max(aabb_query_reduction_percentages, axis=1)

    #save data
    experiment_name = datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H-%M-%S')
    os.makedirs('test_random_zonotope_dim'+experiment_name)
    np.save('test_random_zonotope_dim'+experiment_name+'/voronoi_precomputation_times_median', voronoi_precomputation_times_median)
    np.save('test_random_zonotope_dim'+experiment_name+'/voronoi_precomputation_times_min', voronoi_precomputation_times_min)
    np.save('test_random_zonotope_dim'+experiment_name+'/voronoi_precomputation_times_max', voronoi_precomputation_times_max)

    np.save('test_random_zonotope_dim'+experiment_name+'/voronoi_query_times_median', voronoi_query_times_median)
    np.save('test_random_zonotope_dim'+experiment_name+'/voronoi_query_times_min', voronoi_query_times_min)
    np.save('test_random_zonotope_dim'+experiment_name+'/voronoi_query_times_max', voronoi_query_times_max)
    np.save('test_random_zonotope_dim'+experiment_name+'/voronoi_query_reduction_percentages_median', voronoi_query_reduction_percentages_median)
    np.save('test_random_zonotope_dim'+experiment_name+'/voronoi_query_reduction_percentages_min', voronoi_query_reduction_percentages_min)
    np.save('test_random_zonotope_dim'+experiment_name+'/voronoi_query_reduction_percentages_max', voronoi_query_reduction_percentages_max)

    np.save('test_random_zonotope_dim'+experiment_name+'/aabb_precomputation_times_median', aabb_precomputation_times_median)
    np.save('test_random_zonotope_dim'+experiment_name+'/aabb_precomputation_times_min', aabb_precomputation_times_min)
    np.save('test_random_zonotope_dim'+experiment_name+'/aabb_precomputation_times_max', aabb_precomputation_times_max)

    np.save('test_random_zonotope_dim'+experiment_name+'/aabb_query_times_median', aabb_query_times_median)
    np.save('test_random_zonotope_dim'+experiment_name+'/aabb_query_times_min', aabb_query_times_min)
    np.save('test_random_zonotope_dim'+experiment_name+'/aabb_query_times_max', aabb_query_times_max)

    np.save('test_random_zonotope_dim'+experiment_name+'/aabb_query_reduction_percentages_median', aabb_query_reduction_percentages_median)
    np.save('test_random_zonotope_dim'+experiment_name+'/aabb_query_reduction_percentages_min', aabb_query_reduction_percentages_min)
    np.save('test_random_zonotope_dim'+experiment_name+'/aabb_query_reduction_percentages_max', aabb_query_reduction_percentages_max)

    params = np.array([['dim', np.atleast_1d(dims)],['count', np.atleast_1d(count)], ['construction_repeats', np.atleast_1d(construction_repeats)], \
                       ['queries', np.atleast_1d(queries)],['seed',np.atleast_1d(seed)], ['random_zonotope_generator', random_zonotope_generator.__name__]])
    np.save('test_random_zonotope_dim'+experiment_name+'/params', params)

    # plots
    fig_index = 0
    plt.figure(fig_index)
    fig_index += 1
    plt.subplot(111)
    plt.errorbar(dims, voronoi_precomputation_times_median, np.vstack([voronoi_precomputation_times_median-voronoi_precomputation_times_min,
                                                                       voronoi_precomputation_times_max-voronoi_precomputation_times_median]), marker='.', color = 'b', ecolor='b', elinewidth=0.3,
                 capsize=2, linewidth=0.5, markersize=7)
    plt.errorbar(dims, aabb_precomputation_times_median, np.vstack([aabb_precomputation_times_median-aabb_precomputation_times_min,
                                                                    aabb_precomputation_times_max-aabb_precomputation_times_median]), marker='.', color='r', ecolor='r', elinewidth=0.3,
                 capsize=2, linewidth=0.5, markersize=7)
    plt.legend(['Triangle Ineq.', 'AABB'])
    plt.xlabel('State Dimension')
    plt.ylabel('Precomputation Time (s)')
    plt.title('Test on Uniform Synthetic Dataset')
    #
    # plt.subplot(212)
    # plt.plot(dims, np.log(voronoi_precomputation_times_median))
    # plt.xlabel('$log$ State Dimension')
    # plt.ylabel('$log$ Precomputation Time (s)')
    # # plt.title('Voronoi Closest Zonotope Precomputation Time with %d Zonotopes' %count)
    # plt.tight_layout()

    if save:
        plt.savefig('precomputation_time' + str(default_timer()) + '.png', dpi=500)

    plt.figure(fig_index)
    fig_index += 1
    plt.subplot(111)
    plt.errorbar(dims, voronoi_query_times_median, np.vstack([voronoi_query_times_median-voronoi_query_times_min,
                                                              voronoi_query_times_max-voronoi_query_times_median]), marker='.', color='b', ecolor='b', elinewidth=0.3, capsize=2,
                 linewidth=0.5, markersize=7)
    plt.errorbar(dims, aabb_query_times_median, np.vstack([aabb_query_times_median-aabb_query_times_min,
                                                           aabb_query_times_max-aabb_query_times_min]), marker='.', color='r', ecolor='r', elinewidth=0.3, capsize=2,
                 linewidth=0.5, markersize=7)
    plt.legend(['Triangle Ineq.', 'AABB'])
    plt.xlabel('State Dimension')
    plt.ylabel('Query Time (s)')
    plt.title('Test on Uniform Synthetic Dataset')

    # plt.subplot(212)
    # plt.plot(np.log(dims), np.log(voronoi_query_times_median))
    # plt.xlabel('$log$ State Dimension')
    # plt.ylabel('$log$ Query Time (s)')
    # # plt.title('$log$ Voronoi Closest Zonotope Single Query Time with %d Zonotopes' %count)
    # plt.tight_layout()

    if save:
        plt.savefig('query_time' + str(default_timer()) + '.png', dpi=500)

    plt.figure(fig_index)
    fig_index += 1
    plt.subplot(111)
    plt.errorbar(dims, voronoi_query_reduction_percentages_median, np.vstack([voronoi_query_reduction_percentages_median-voronoi_query_reduction_percentages_min,
                                                                              voronoi_query_reduction_percentages_max - voronoi_query_reduction_percentages_median]), marker='.', color='b',ecolor='b',
                 elinewidth=0.3, capsize=2, linewidth=0.5, markersize=7)
    plt.errorbar(dims, aabb_query_reduction_percentages_median, np.vstack([aabb_query_reduction_percentages_median-aabb_query_reduction_percentages_min,
                                                                           aabb_query_reduction_percentages_max-aabb_query_reduction_percentages_median]), marker='.', color='r', ecolor='r',
                 elinewidth=0.3, capsize=2, linewidth=0.5, markersize=7)
    plt.legend(['Triangle Ineq.', 'AABB'])
    plt.xlabel('State Dimension')
    plt.ylabel('% of Polytopes Evaluated')
    plt.title('Test on Uniform Synthetic Dataset')

    # plt.subplot(212)
    # plt.plot(np.log(dims), np.log(voronoi_query_reduction_percentages_median))
    # plt.xlabel('$log$ State Dimension')
    # plt.ylabel('$log$ % of Zonotopes Evaluated')
    # # plt.title('$log$ Voronoi Closest Zonotope Reduction Percentage with %d Zonotopes' %count)
    # plt.tight_layout()

    if save:
        plt.savefig('reduction_percentage' + str(default_timer()) + '.png', dpi=500)
    else:
        plt.show()

    plt.figure(fig_index)
    fig_index += 1
    plt.subplot(111)
    plt.errorbar(dims, count/100*voronoi_query_reduction_percentages_median, count/100*np.vstack([voronoi_query_reduction_percentages_median-voronoi_query_reduction_percentages_min,
                                                                              voronoi_query_reduction_percentages_max - voronoi_query_reduction_percentages_median]), marker='.', color='b',ecolor='b',
                 elinewidth=0.3, capsize=2, linewidth=0.5, markersize=7)
    plt.errorbar(dims, count/100*aabb_query_reduction_percentages_median, count/100*np.vstack([aabb_query_reduction_percentages_median-aabb_query_reduction_percentages_min,
                                                                           aabb_query_reduction_percentages_max-aabb_query_reduction_percentages_median]), marker='.', color='r', ecolor='r',
                 elinewidth=0.3, capsize=2, linewidth=0.5, markersize=7)
    plt.legend(['Triangle Ineq.', 'AABB'])
    plt.xlabel('State Dimension')
    plt.ylabel('Number of Zonotopes Evaluated')
    plt.title('Test on Uniform Synthetic Dataset')

    # plt.subplot(212)
    # plt.plot(np.log(dims), np.log(voronoi_query_reduction_percentages_median))
    # plt.xlabel('$log$ State Dimension')
    # plt.ylabel('$log$ % of Zonotopes Evaluated')
    # # plt.title('$log$ Voronoi Closest Zonotope Reduction Percentage with %d Zonotopes' %count)
    # plt.tight_layout()

    if save:
        plt.savefig('number_evaluated' + str(default_timer()) + '.png', dpi=500)
    else:
        plt.show()


def find_extremum(polytopes, dim):
    from bounding_box.box import AH_polytope_to_box
    maxs = np.ones(dim)*(-np.inf)
    mins = np.ones(dim) * (np.inf)
    for p in polytopes:
        lu = AH_polytope_to_box(p)
        maxs = np.maximum(lu[dim:], maxs)
        mins = np.minimum(lu[0:dim], mins)
    print((maxs, mins))
    return np.vstack([mins, maxs])


def test_on_rrt(dir, queries, query_range):
    polytope_sets, times = get_polytope_sets_in_dir(dir)
    polytope_counts = np.asarray([len(p) for p in polytope_sets])
    # print(polytope_counts)
    voronoi_precomputation_times = np.zeros([len(times)])
    voronoi_query_times = np.zeros([len(times), queries])
    voronoi_query_reduction_percentages = np.zeros([len(times),queries])
    aabb_precomputation_times = np.zeros([len(times)])
    aabb_query_times = np.zeros([len(times), queries])
    aabb_query_reduction_percentages = np.zeros([len(times),queries])

    if query_range is None:
        if query_range is None:
            query_range = find_extremum(polytope_sets[-1], 10).T
            query_avg = (query_range[:, 1] + query_range[:, 0]) / 2
            query_diff = (query_range[:, 1] - query_range[:, 0]) / 2 * 1.05
        else:
            query_avg = (query_range[:, 1] + query_range[:, 0]) / 2
            query_diff = (query_range[:, 1] - query_range[:, 0]) / 2
    else:
        query_avg = (query_range[:,1]+query_range[:,0])/2
        query_diff = (query_range[:,1]-query_range[:,0])/2

    for i, polytopes in enumerate(polytope_sets):
        # test voronoi
        print(('Length of polytopes is %i' %len(polytopes)))
        construction_start_time = default_timer()
        print('Precomputing TI...')
        max_number_key_points = int(1000000/len(polytopes))
        # max_number_key_points=None
        # print('keypoint limit is %i' % max_number_key_points)
        vcp = VoronoiClosestPolytope(polytopes, max_number_key_points=max_number_key_points)
        voronoi_precomputation_times[i] = default_timer() - construction_start_time
        print(('TI Precomputation completed in %f s!' %voronoi_precomputation_times[i]))
        # query
        print('Querying TI...')
        for query_index in range(queries):
            query_point = np.multiply((np.random.rand(query_avg.shape[0]) - 0.5) *2, query_diff)+query_avg
            query_start_time = default_timer()
            best_zonotope, best_distance, evaluated_zonotopes = vcp.find_closest_polytope(query_point,
                                                                                          return_intermediate_info=True)
            voronoi_query_times[i, query_index] = default_timer() - query_start_time
            voronoi_query_reduction_percentages[i, query_index] = len(evaluated_zonotopes) * 100. / len(polytopes)

        print('TI Querying completed!')

        # test aabb
        construction_start_time = default_timer()
        print('Precomputing AABB...')
        zono_tree = PolytopeTree(polytopes)
        aabb_precomputation_times[i] = default_timer() - construction_start_time
        print(('AABB Precomputation completed in %f s!' % aabb_precomputation_times[i]))
        # query
        print('Querying AABB...')
        for query_index in range(queries):
            query_point = np.multiply((np.random.rand(query_avg.shape[0]) - 0.5) *2, query_diff)+query_avg
            query_start_time = default_timer()
            best_zonotope, best_distance, evaluated_zonotopes, query_box = zono_tree.find_closest_polytopes(
                query_point, return_intermediate_info=True)
            aabb_query_times[i, query_index] = default_timer() - query_start_time
            aabb_query_reduction_percentages[i, query_index] = len(evaluated_zonotopes) * 100. / len(polytopes)
        print('AABB Querying completed!')

        voronoi_precomputation_times_avg = voronoi_precomputation_times

        voronoi_query_times_median = np.median(voronoi_query_times, axis=1)
        voronoi_query_times_min = np.min(voronoi_query_times, axis=1)
        voronoi_query_times_max = np.max(voronoi_query_times, axis=1)

        voronoi_query_reduction_percentages_median = np.median(voronoi_query_reduction_percentages, axis=1)
        voronoi_query_reduction_percentages_min = np.min(voronoi_query_reduction_percentages, axis=1)
        voronoi_query_reduction_percentages_max = np.max(voronoi_query_reduction_percentages, axis=1)

        aabb_precomputation_times_avg = aabb_precomputation_times

        aabb_query_times_median = np.median(aabb_query_times, axis=1)
        aabb_query_times_min = np.min(aabb_query_times, axis=1)
        aabb_query_times_max = np.max(aabb_query_times, axis=1)
        aabb_query_reduction_percentages_median = np.median(aabb_query_reduction_percentages, axis=1)
        aabb_query_reduction_percentages_min = np.min(aabb_query_reduction_percentages, axis=1)
        aabb_query_reduction_percentages_max = np.max(aabb_query_reduction_percentages, axis=1)
        #save files
        # save data
    # print(aabb_query_times_min)
    experiment_name = datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H-%M-%S')
    os.makedirs('test_on_rrt' + experiment_name)
    np.save('test_on_rrt' + experiment_name + '/voronoi_precomputation_times_median',
            voronoi_precomputation_times_avg)
    np.save('test_on_rrt' + experiment_name + '/voronoi_query_times_median', voronoi_query_times_median)
    np.save('test_on_rrt' + experiment_name + '/voronoi_query_times_min', voronoi_query_times_min)
    np.save('test_on_rrt' + experiment_name + '/voronoi_query_times_max', voronoi_query_times_max)
    np.save('test_on_rrt' + experiment_name + '/voronoi_query_reduction_percentages_median',
            voronoi_query_reduction_percentages_median)
    np.save('test_on_rrt' + experiment_name + '/voronoi_query_reduction_percentages_min',
            voronoi_query_reduction_percentages_min)

    np.save('test_on_rrt' + experiment_name + '/voronoi_query_reduction_percentages_max',
            voronoi_query_reduction_percentages_max)

    np.save('test_on_rrt' + experiment_name + '/aabb_precomputation_times_median',
            aabb_precomputation_times_avg)
    np.save('test_on_rrt' + experiment_name + '/aabb_query_times_median', aabb_query_times_median)
    np.save('test_on_rrt' + experiment_name + '/aabb_query_times_min', aabb_query_times_min)
    np.save('test_on_rrt' + experiment_name + '/aabb_query_times_max', aabb_query_times_max)

    np.save('test_on_rrt' + experiment_name + '/aabb_query_reduction_percentages_median',
            aabb_query_reduction_percentages_median)
    np.save('test_on_rrt' + experiment_name + '/aabb_query_reduction_percentages_min',
            aabb_query_reduction_percentages_min)
    np.save('test_on_rrt' + experiment_name + '/aabb_query_reduction_percentages_max',
            aabb_query_reduction_percentages_max)

    params = np.array([['dim', np.atleast_1d(len(times))],
                       ['construction_repeats', np.atleast_1d(1)], \
                       ['queries', np.atleast_1d(queries)]])
    np.save('test_on_rrt' + experiment_name + '/times', times)
    np.save('test_on_rrt' + experiment_name + '/params', params)
    np.save('test_on_rrt' + experiment_name + '/polytope_counts', polytope_counts)
    # plots
    fig_index = 0
    plt.figure(fig_index)
    fig_index += 1
    plt.subplot(111)
    plt.plot(polytope_counts, voronoi_precomputation_times_avg, marker='.', color = 'b',
                 linewidth=0.5, markersize=7)
    plt.plot(polytope_counts, aabb_precomputation_times_avg, marker='.', color='r',linewidth=0.5, markersize=7)
    plt.legend(['Triangle Ineq.', 'AABB'])
    plt.xlabel('Number of Polytopes')
    plt.ylabel('Precomputation Time (s)')
    plt.title('Closest Zonotope Precomputation Time with 2D Hopper Dataset')
    #
    # plt.subplot(212)
    # plt.plot(dims, np.log(voronoi_precomputation_times_median))
    # plt.xlabel('$log$ State Dimension')
    # plt.ylabel('$log$ Precomputation Time (s)')
    # # plt.title('Voronoi Closest Zonotope Precomputation Time with %d Zonotopes' %count)
    # plt.tight_layout()

    plt.savefig('test_on_rrt' + experiment_name + '/precomputation_time' + str(default_timer()) + '.png', dpi=500)

    plt.figure(fig_index)
    fig_index += 1
    plt.subplot(111)
    # plt.errorbar(polytope_counts, voronoi_query_times_median, voronoi_query_times_std, marker='.', color='b', ecolor='b', elinewidth=0.3, capsize=2,
    #              linewidth=0.5, markersize=7)
    # plt.errorbar(polytope_counts, aabb_query_times_median, aabb_query_times_std, marker='.', color='r', ecolor='r', elinewidth=0.3, capsize=2,
    #              linewidth=0.5, markersize=7)
    plt.plot(polytope_counts, voronoi_query_times_median, marker='.', color='b',
                 linewidth=0.5, markersize=7)
    plt.plot(polytope_counts, aabb_query_times_median, marker='.', color='r',
                 linewidth=0.5, markersize=7)

    plt.legend(['Triangle Ineq.', 'AABB'])
    plt.xlabel('Number of Polytopes')
    plt.ylabel('Query Time (s)')
    plt.title('Closest Zonotope Single Query Time with 2D Hopper Dataset')

    # plt.subplot(212)
    # plt.plot(np.log(dims), np.log(voronoi_query_times_median))
    # plt.xlabel('$log$ State Dimension')
    # plt.ylabel('$log$ Query Time (s)')
    # # plt.title('$log$ Voronoi Closest Zonotope Single Query Time with %d Zonotopes' %count)
    # plt.tight_layout()

    plt.savefig('test_on_rrt' + experiment_name + '/query_time' + str(default_timer()) + '.png', dpi=500)

    plt.figure(fig_index)
    fig_index += 1
    plt.subplot(111)
    plt.plot(polytope_counts, voronoi_query_reduction_percentages_median, marker='.', color='b',
                 linewidth=0.5, markersize=7)
    plt.plot(polytope_counts, aabb_query_reduction_percentages_median, marker='.', color='r',
                 linewidth=0.5, markersize=7)

    # plt.errorbar(polytope_counts, voronoi_query_reduction_percentages_median, voronoi_query_reduction_percentages_std, marker='.', color='b',ecolor='b',
    #              elinewidth=0.3, capsize=2, linewidth=0.5, markersize=7)
    # plt.errorbar(polytope_counts, aabb_query_reduction_percentages_median, aabb_query_reduction_percentages_std, marker='.', color='r', ecolor='r',
    #              elinewidth=0.3, capsize=2, linewidth=0.5, markersize=7)
    plt.legend(['Triangle Ineq.', 'AABB'])
    plt.xlabel('Number of polytopes')
    plt.ylabel('% of Polytopes Evaluated')
    plt.title('Nearest Polytope Reduction Percentage with 2D Hopper Dataset')
    plt.savefig('test_on_rrt' + experiment_name + '/reduction_percentage' + str(default_timer()) + '.png', dpi=500)

def test_on_mpc(dir, queries, query_range):
    polytope_sets = get_polytope_sets_in_dir(dir, data_source='mpc')
    polytope_counts = np.asarray([len(p) for p in polytope_sets])
    print(polytope_counts)
    # extract zonotopes
    # print(polytope_counts)
    voronoi_precomputation_times = np.zeros([len(polytope_sets)])
    voronoi_query_times = np.zeros([len(polytope_sets), queries])
    voronoi_query_reduction_percentages = np.zeros([len(polytope_sets),queries])
    aabb_precomputation_times = np.zeros([len(polytope_sets)])
    aabb_query_times = np.zeros([len(polytope_sets), queries])
    aabb_query_reduction_percentages = np.zeros([len(polytope_sets), queries])
    all_xs = []
    for p in polytope_sets[-1]:
        all_xs.append(np.ndarray.flatten(p.x))
    all_xs = np.asarray(all_xs)
    if query_range is None:
        query_range = np.vstack([np.min(all_xs, axis=0), np.max(all_xs, axis=0)]).T
        query_avg = (query_range[:,1]+query_range[:,0])/2
        query_diff = (query_range[:,1]-query_range[:,0])/2*1.05
    else:
        query_avg = (query_range[:,1]+query_range[:,0])/2
        query_diff = (query_range[:,1]-query_range[:,0])/2
    for i, polytopes in enumerate(polytope_sets):
        # test voronoi
        construction_start_time = default_timer()
        print('Precomputing TI...')
        max_number_key_points = 1000000/len(polytopes)
        print(('keypoint limit is %i' % max_number_key_points))
        vcp = VoronoiClosestPolytope(polytopes, max_number_key_points=max_number_key_points)
        voronoi_precomputation_times[i] = default_timer() - construction_start_time
        print(('TI Precomputation completed in %f s!' %voronoi_precomputation_times[i]))
        # query
        print('Querying TI...')
        for query_index in range(queries):
            query_point = np.multiply((np.random.rand(query_avg.shape[0]) - 0.5) *2, query_diff)+query_avg
            query_start_time = default_timer()
            best_zonotope, best_distance, evaluated_zonotopes = vcp.find_closest_polytope(query_point,
                                                                                          return_intermediate_info=True)
            voronoi_query_times[i, query_index] = default_timer() - query_start_time
            voronoi_query_reduction_percentages[i, query_index] = len(evaluated_zonotopes) * 100. / len(polytopes)

        print('TI Querying completed!')

        # test aabb
        construction_start_time = default_timer()
        print('Precomputing AABB...')
        zono_tree = PolytopeTree(polytopes)
        aabb_precomputation_times[i] = default_timer() - construction_start_time
        print(('AABB Precomputation completed in %f s!' % aabb_precomputation_times[i]))
        # query
        print('Querying AABB...')
        for query_index in range(queries):
            query_point = np.multiply((np.random.rand(query_avg.shape[0]) - 0.5) *2, query_diff)+query_avg
            query_start_time = default_timer()
            best_zonotope, best_distance, evaluated_zonotopes, query_box = zono_tree.find_closest_polytopes(
                query_point, return_intermediate_info=True)
            aabb_query_times[i, query_index] = default_timer() - query_start_time
            aabb_query_reduction_percentages[i, query_index] = len(evaluated_zonotopes) * 100. / len(polytopes)
        print('AABB Querying completed!')

        voronoi_precomputation_times_avg = voronoi_precomputation_times

        voronoi_query_times_median = np.median(voronoi_query_times, axis=1)
        voronoi_query_times_min = np.min(voronoi_query_times, axis=1)
        voronoi_query_times_max = np.max(voronoi_query_times, axis=1)

        voronoi_query_reduction_percentages_median = np.median(voronoi_query_reduction_percentages, axis=1)
        voronoi_query_reduction_percentages_min = np.min(voronoi_query_reduction_percentages, axis=1)
        voronoi_query_reduction_percentages_max = np.max(voronoi_query_reduction_percentages, axis=1)

        aabb_precomputation_times_avg = aabb_precomputation_times

        aabb_query_times_median = np.median(aabb_query_times, axis=1)
        aabb_query_times_min = np.min(aabb_query_times, axis=1)
        aabb_query_times_max = np.max(aabb_query_times, axis=1)
        aabb_query_reduction_percentages_median = np.median(aabb_query_reduction_percentages, axis=1)
        aabb_query_reduction_percentages_min = np.min(aabb_query_reduction_percentages, axis=1)
        aabb_query_reduction_percentages_max = np.max(aabb_query_reduction_percentages, axis=1)
        #save files
        # save data
    # print(aabb_query_times_min)
    experiment_name = datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H-%M-%S')
    os.makedirs('test_on_mpc' + experiment_name)
    np.save('test_on_mpc' + experiment_name + '/voronoi_precomputation_times_median',
            voronoi_precomputation_times_avg)
    np.save('test_on_mpc' + experiment_name + '/voronoi_query_times_median', voronoi_query_times_median)
    np.save('test_on_mpc' + experiment_name + '/voronoi_query_times_min', voronoi_query_times_min)
    np.save('test_on_mpc' + experiment_name + '/voronoi_query_times_max', voronoi_query_times_max)
    np.save('test_on_mpc' + experiment_name + '/voronoi_query_reduction_percentages_median',
            voronoi_query_reduction_percentages_median)
    np.save('test_on_mpc' + experiment_name + '/voronoi_query_reduction_percentages_min',
            voronoi_query_reduction_percentages_min)

    np.save('test_on_mpc' + experiment_name + '/voronoi_query_reduction_percentages_max',
            voronoi_query_reduction_percentages_max)

    np.save('test_on_mpc' + experiment_name + '/aabb_precomputation_times_median',
            aabb_precomputation_times_avg)
    np.save('test_on_mpc' + experiment_name + '/aabb_query_times_median', aabb_query_times_median)
    np.save('test_on_mpc' + experiment_name + '/aabb_query_times_min', aabb_query_times_min)
    np.save('test_on_mpc' + experiment_name + '/aabb_query_times_max', aabb_query_times_max)

    np.save('test_on_mpc' + experiment_name + '/aabb_query_reduction_percentages_median',
            aabb_query_reduction_percentages_median)
    np.save('test_on_mpc' + experiment_name + '/aabb_query_reduction_percentages_min',
            aabb_query_reduction_percentages_min)
    np.save('test_on_mpc' + experiment_name + '/aabb_query_reduction_percentages_max',
            aabb_query_reduction_percentages_max)

    params = np.array([['dim', np.atleast_1d(len(polytope_sets))],
                       ['construction_repeats', np.atleast_1d(1)], \
                       ['queries', np.atleast_1d(queries)]])
    # np.save('test_on_mpc' + experiment_name + '/times', times)
    np.save('test_on_mpc' + experiment_name + '/params', params)
    np.save('test_on_mpc' + experiment_name + '/polytope_counts', polytope_counts)
    # plots
    fig_index = 0
    plt.figure(fig_index)
    fig_index += 1
    plt.subplot(111)
    plt.plot(polytope_counts, voronoi_precomputation_times_avg, marker='.', color = 'b',
                 linewidth=0.5, markersize=7)
    plt.plot(polytope_counts, aabb_precomputation_times_avg, marker='.', color='r',linewidth=0.5, markersize=7)
    plt.legend(['Triangle Ineq.', 'AABB'])
    plt.xlabel('Number of Polytopes')
    plt.ylabel('Precomputation Time (s)')
    plt.title('Closest Zonotope Precomputation Time with 2D Hopper Dataset')
    #
    # plt.subplot(212)
    # plt.plot(dims, np.log(voronoi_precomputation_times_median))
    # plt.xlabel('$log$ State Dimension')
    # plt.ylabel('$log$ Precomputation Time (s)')
    # # plt.title('Voronoi Closest Zonotope Precomputation Time with %d Zonotopes' %count)
    # plt.tight_layout()

    plt.savefig('test_on_mpc' + experiment_name + '/precomputation_time' + str(default_timer()) + '.png', dpi=500)

    plt.figure(fig_index)
    fig_index += 1
    plt.subplot(111)
    # plt.errorbar(polytope_counts, voronoi_query_times_median, voronoi_query_times_std, marker='.', color='b', ecolor='b', elinewidth=0.3, capsize=2,
    #              linewidth=0.5, markersize=7)
    # plt.errorbar(polytope_counts, aabb_query_times_median, aabb_query_times_std, marker='.', color='r', ecolor='r', elinewidth=0.3, capsize=2,
    #              linewidth=0.5, markersize=7)
    plt.plot(polytope_counts, voronoi_query_times_median, marker='.', color='b',
                 linewidth=0.5, markersize=7)
    plt.plot(polytope_counts, aabb_query_times_median, marker='.', color='r',
                 linewidth=0.5, markersize=7)

    plt.legend(['Triangle Ineq.', 'AABB'])
    plt.xlabel('Number of Polytopes')
    plt.ylabel('Query Time (s)')
    plt.title('Closest Zonotope Single Query Time with 2D Hopper Dataset')

    # plt.subplot(212)
    # plt.plot(np.log(dims), np.log(voronoi_query_times_median))
    # plt.xlabel('$log$ State Dimension')
    # plt.ylabel('$log$ Query Time (s)')
    # # plt.title('$log$ Voronoi Closest Zonotope Single Query Time with %d Zonotopes' %count)
    # plt.tight_layout()

    plt.savefig('test_on_mpc' + experiment_name + '/query_time' + str(default_timer()) + '.png', dpi=500)

    plt.figure(fig_index)
    fig_index += 1
    plt.subplot(111)
    plt.plot(polytope_counts, voronoi_query_reduction_percentages_median, marker='.', color='b',
                 linewidth=0.5, markersize=7)
    plt.plot(polytope_counts, aabb_query_reduction_percentages_median, marker='.', color='r',
                 linewidth=0.5, markersize=7)

    # plt.errorbar(polytope_counts, voronoi_query_reduction_percentages_median, voronoi_query_reduction_percentages_std, marker='.', color='b',ecolor='b',
    #              elinewidth=0.3, capsize=2, linewidth=0.5, markersize=7)
    # plt.errorbar(polytope_counts, aabb_query_reduction_percentages_median, aabb_query_reduction_percentages_std, marker='.', color='r', ecolor='r',
    #              elinewidth=0.3, capsize=2, linewidth=0.5, markersize=7)
    plt.legend(['Triangle Ineq.', 'AABB'])
    plt.xlabel('Number of polytopes')
    plt.ylabel('% of Polytopes Evaluated')
    plt.title('Nearest Polytope Reduction Percentage with 2D Hopper Dataset')
    plt.savefig('test_on_mpc' + experiment_name + '/reduction_percentage' + str(default_timer()) + '.png', dpi=500)



if __name__ == '__main__':
    # print('time_against_count(dim=6, counts=np.arange(2, 11, 2) * 100, construction_repeats=3, queries=1000), random_zonotope_generator=get_uniform_random_zonotopes')
    # test_random_zonotope_count(dim=6, counts=np.arange(2, 11, 1) * 100, construction_repeats=3, queries=1000, random_zonotope_generator=get_line_random_zonotopes, save=True)
    # print('test_uniform_random_zonotope_dim(count=500, dims=np.arange(2, 11, 1), construction_repeats=3, queries=100), random_zonotope_generator=get_line_random_zonotopes')
    # test_random_zonotope_dim(count=500, dims=np.arange(2, 11, 1), construction_repeats=3, queries=1000, random_zonotope_generator=get_line_random_zonotopes)
    #
    # test_voronoi_closest_zonotope(100, save=False)
    # For pendulum
    # test_on_rrt('/Users/albertwu/Google Drive/MIT/RobotLocomotion/Closest Polytope/ACC2020/Datasets/R3T_Pendulum_20190919_21-59-04', queries=1000, query_range=np.asarray([[-4, 4],[-13,13]]))
    # For hopper
    # test_on_rrt('/Users/albertwu/Google Drive/MIT/RobotLocomotion/Closest Polytope/ACC2020/Datasets/RRT_Hopper_2d_20190919_22-00-37', queries=1000, query_range=np.asarray([[-15, 25],[-1,2.5],[-np.pi/2,np.pi/2],[-np.pi/3,np.pi/3],[2,6],\
    #                                                                                                            [-2,2],[-10,10],[-5,5],[-3,3],[-10,10]]))
    test_on_rrt('/Users/albertwu/Google Drive/MIT/RobotLocomotion/Closest Polytope/ACC2020/Datasets/RRT_Hopper_2d_20190919_22-00-37', queries=1000, query_range=None)
    # test_on_rrt(
    #     '/Users/albertwu/Google Drive/MIT/RobotLocomotion/Closest Polytope/ACC2020/Datasets/RRT_Hopper_2d_20190919_22-00-37',
    #     queries=1000,
    #     query_range=None)

    # For mpc
    # Pendulum
    # test_on_mpc('/Users/albertwu/Google Drive/MIT/RobotLocomotion/Closest Polytope/ACC2020/Datasets/MPC', queries=1000, query_range=np.asarray([[-0.135, 0.135],[-1.1,1.1]]))
    # Bar balancing
    # test_on_mpc('/Users/albertwu/Google Drive/MIT/RobotLocomotion/Closest Polytope/ACC2020/Datasets/MPC', queries=1000,
    #             query_range=None)
