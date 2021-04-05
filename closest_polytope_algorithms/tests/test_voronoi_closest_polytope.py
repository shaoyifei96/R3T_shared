from voronoi.voronoi import *
from pypolycontain.utils.random_polytope_generator import *
from pypolycontain.visualization.visualize_2D import visualize_2D_zonotopes as visZ
from scipy.spatial import voronoi_plot_2d
from timeit import default_timer
from time import time
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

plt.rcParams["font.family"] = "Times New Roman"


def test_voronoi_closest_zonotope(zonotope_count = 30, seed=None,save=True, key_vertices_count=3):
    AH_polytopes = get_uniform_random_zonotopes(zonotope_count, dim=2, generator_range=zonotope_count*1.5,return_type='zonotope',\
                                             centroid_range=zonotope_count*4, seed=seed)
    zonotopes = get_uniform_random_zonotopes(zonotope_count, dim=2, generator_range=zonotope_count*1.5,return_type='zonotope',\
                                             centroid_range=zonotope_count*4, seed=seed)
    #precompute
    vca = VoronoiClosestPolytope(AH_polytopes, key_vertices_count)
    #build query point
    query_point = (np.random.rand(2)-0.5)*zonotope_count*5
    print(('Query point '+ str(query_point)))
    np.reshape(query_point,(query_point.shape[0],1))

    #query
    closest_polytope, best_distance, closest_AHpolytope_candidates = vca.find_closest_polytope(query_point,return_intermediate_info=True)

    #find indices for plotting
    candidate_indices = np.zeros(len(closest_AHpolytope_candidates), dtype='int')
    closest_index = np.zeros(1, dtype='int')
    for i, cac in enumerate(closest_AHpolytope_candidates):
        for j in range(len(AH_polytopes)):
            if cac == AH_polytopes[j]:
                candidate_indices[i] = j
                break
    for j in range(len(AH_polytopes)):
        if closest_polytope == AH_polytopes[j]:
            closest_index[0] = j
            break
    # print(candidate_indices)

    #visualize voronoi
    # fig = voronoi_plot_2d(vca.centroid_voronoi, point_size=2,show_vertices=False, line_alpha=0.4, line_width=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # dist, voronoi_centroid_index = vca.centroid_tree.query(query_point)

    print(('Checked %d polytopes' %len(closest_AHpolytope_candidates)))
    #visualize vertex circles
    # #sanity check
    # vertex = vca.centroid_voronoi.vertices[15]
    # for centroid_index in vca.vertex_to_voronoi_centroid_index[str(vertex)]:
    #     centroid = vca.centroid_voronoi.points[centroid_index]
    #     plt.scatter(centroid[0], centroid[1], facecolor='green', s=15)
    # plt.scatter(vertex[0], vertex[1], facecolor='green', s=15)

    # voronoi_centroid = vca.centroid_voronoi.points[voronoi_centroid_index]
    # print('Centroid dist '+str(dist))
    # print('Voronoi Centroid index ' + str(voronoi_centroid_index))
    # print('Closest Voronoi centroid ' + str(vca.centroid_tree.data[voronoi_centroid_index,:]))
    # print('Closest polytope centroid ' + str(evaluated_zonotopes[0].x))
    # plt.scatter(voronoi_centroid[0], voronoi_centroid[1], facecolor='blue', s=6)
    #visualize centroids
    for vertex in vca.key_points:
        plt.scatter(vertex[0], vertex[1], facecolor='c', s=2, alpha=1)

    #visualize polytopes

    fig, ax = visZ(zonotopes[candidate_indices], title="", fig=fig, ax=ax, alpha=0.3, color='pink')
    fig, ax = visZ(zonotopes[closest_index], title="", fig=fig, ax=ax, alpha=0.8,color='brown')
    fig, ax = visZ(zonotopes, title="", alpha=0.07, fig=fig, ax=ax, color='gray')
    plt.scatter(query_point[0],query_point[1], facecolor='red', s=6)
    plt.axes().set_aspect('equal')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Closest Zonotope with Voronoi Diagram')
    print(('Closest Zonotope: ', closest_polytope))
    if save:
        plt.savefig('closest_zonotope'+str(default_timer())+'.png', dpi=500)
    else:
        plt.show()

def test_voronoi_closest_zonotope_nd(zonotope_count = 30,dim = 2):
    zonotopes = get_uniform_random_zonotopes(zonotope_count, dim=dim, generator_range=zonotope_count*1.2,return_type='zonotope')
    #precompute
    vca = VoronoiClosestPolytope(zonotopes)
    #build query point
    query_point = (np.random.rand(dim)-0.5)*zonotope_count*3
    np.reshape(query_point,(query_point.shape[0],1))

    #query
    best_polytope, best_distance, evaluated_zonotopes = vca.find_closest_polytope(query_point, return_intermediate_info=True)
    print(('Checked %d of %d zonotopes' %(len(evaluated_zonotopes), zonotope_count)))
    print(('Closest zonotope is ', best_polytope))

def time_against_count(dim=2, counts = np.arange(3, 16, 3)*10, construction_repeats = 1, queries=100,save=True):

    precomputation_times = np.zeros([len(counts), construction_repeats])
    query_times = np.zeros([len(counts), construction_repeats*queries])
    query_reduction_percentages = np.zeros([len(counts), construction_repeats*queries])

    for cr_index in range(construction_repeats):
        print(('Repetition %d' %cr_index))
        for count_index, count in enumerate(counts):
            print(('Testing %d zonotopes...' % count))
            zonotopes = get_uniform_random_zonotopes(count, dim=dim, generator_range=count * 1.2,return_type='zonotope')
            construction_start_time = default_timer()
            vcp = VoronoiClosestPolytope(zonotopes)
            precomputation_times[count_index, cr_index] = default_timer()-construction_start_time
            #query
            for query_index in range(queries):
                query_point = (np.random.rand(dim) - 0.5) * count * 5 #random query point
                query_start_time = default_timer()
                best_zonotope, best_distance, evaluated_zonotopes = vcp.find_closest_polytope(query_point, return_intermediate_info=True)
                query_times[count_index,cr_index*queries+query_index] = default_timer()-query_start_time
                query_reduction_percentages[count_index, cr_index*queries+query_index] = len(evaluated_zonotopes)*100./count

    precomputation_times_avg = np.mean(precomputation_times, axis=1)
    precomputation_times_std = np.std(precomputation_times, axis=1)

    query_times_avg = np.mean(query_times, axis=1)
    query_times_std = np.std(query_times, axis=1)

    query_reduction_percentages_avg =np.mean(query_reduction_percentages, axis=1)
    query_reduction_percentages_std = np.std(query_reduction_percentages, axis=1)

    #plots
    fig_index = 0
    plt.figure(fig_index)
    fig_index+=1
    plt.subplot(211)
    plt.errorbar(counts,precomputation_times_avg,precomputation_times_std,marker='.',ecolor='r',elinewidth=0.3,capsize=2,linewidth=0.5,markersize=7)
    plt.xlabel('Zonotope Count')
    plt.ylabel('Precomputation Time (s)')
    plt.title('Voronoi Closest Zonotope Precomputation Time in %d-D' %dim)

    plt.subplot(212)
    plt.plot(np.log(counts),np.log(precomputation_times_avg))
    plt.xlabel('$log$ Zonotope Count')
    plt.ylabel('$log$ Precomputation Time (s)')
    plt.title('$log$ Voronoi Closest Zonotope Precomputation Time in %d-D' %dim)
    plt.tight_layout()
    if save:
        plt.savefig('precomputation_time' + str(default_timer()) + '.png', dpi=500)

    plt.figure(fig_index)
    fig_index+=1
    plt.subplot(211)
    plt.errorbar(counts,query_times_avg,query_times_std,marker='.',ecolor='r',elinewidth=0.3,capsize=2,linewidth=0.5,markersize=7)
    plt.xlabel('Zonotope Count')
    plt.ylabel('Query Time (s)')
    plt.title('Voronoi Closest Zonotope Single Query Time in %d-D' %dim)

    plt.subplot(212)
    plt.plot(np.log(counts),np.log(query_times_avg))
    plt.xlabel('$log$ Zonotope Count')
    plt.ylabel('$log$ Query Time (s)')
    plt.title('$log$ Voronoi Closest Zonotope Single Query Time in %d-D' %dim)
    plt.tight_layout()
    if save:
        plt.savefig('query_time' + str(default_timer()) + '.png', dpi=500)

    plt.figure(fig_index)
    fig_index+=1
    plt.subplot(211)
    plt.errorbar(counts,query_reduction_percentages_avg,query_reduction_percentages_std,marker='.',ecolor='r',elinewidth=0.3,capsize=2,linewidth=0.5,markersize=7)
    plt.xlabel('Zonotope Count')
    plt.ylabel('% of Zonotopes Evaluated')
    plt.title('Voronoi Closest Zonotope Reduction Percentage in %d-D' %dim)

    plt.subplot(212)
    plt.plot(np.log(counts),np.log(query_reduction_percentages_avg))
    plt.xlabel('$log$ Zonotope Count')
    plt.ylabel('$log$ % of Zonotopes Evaluated')
    plt.title('$log$ Voronoi Closest Zonotope Reduction Percentage in %d-D' %dim)
    plt.tight_layout()
    if save:
        plt.savefig('reduction_percentage' + str(default_timer()) + '.png', dpi=500)

    else:
        plt.show()

def time_against_dim(count = 100, dims = np.arange(2, 11, 1),construction_repeats = 1, queries=100,save=True, process_count = 8):
    precomputation_times = np.zeros([len(dims), construction_repeats])
    query_times = np.zeros([len(dims), construction_repeats*queries])
    query_reduction_percentages = np.zeros([len(dims), construction_repeats*queries])
    for cr_index in range(construction_repeats):
        print(('Repetition %d' %cr_index))
        for dim_index, dim in enumerate(dims):
            print(('Testing zonotopes in %d-D...' % dim))
            # zonotopes = get_uniform_random_zonotopes(count, dim=dim, generator_range=1,centroid_range=count*5, return_type='zonotope', process_count=process_count)
            zonotopes = get_uniform_density_random_polytopes(3, dim=dim, generator_range=1,centroid_range=5, return_type='zonotope', process_count=process_count)
            construction_start_time = default_timer()
            vcp = VoronoiClosestPolytope(zonotopes, process_count)
            precomputation_times[dim_index, cr_index] = default_timer()-construction_start_time
            #query
            for query_index in range(queries):
                query_point = (np.random.rand(dim) - 0.5) * count*2 #random query point
                query_start_time = default_timer()
                best_zonotope, best_distance, evaluated_zonotopes = vcp.find_closest_polytope(query_point, return_intermediate_info=True)
                query_times[dim_index,cr_index*queries+query_index] = default_timer()-query_start_time
                query_reduction_percentages[dim_index, cr_index*queries+query_index] = len(evaluated_zonotopes)*100./count

    precomputation_times_avg = np.mean(precomputation_times, axis=1)
    precomputation_times_std = np.std(precomputation_times, axis=1)

    query_times_avg = np.mean(query_times, axis=1)
    query_times_std = np.std(query_times, axis=1)


    query_reduction_percentages_avg =np.mean(query_reduction_percentages, axis=1)
    query_reduction_percentages_std = np.std(query_reduction_percentages, axis=1)

    #plots
    fig_index = 0
    plt.figure(fig_index)
    fig_index+=1
    plt.subplot(211)
    plt.errorbar(dims,precomputation_times_avg,precomputation_times_std,marker='.',ecolor='r',elinewidth=0.3,capsize=2,linewidth=0.5,markersize=7)
    plt.xlabel('State Dimension')
    plt.ylabel('Precomputation Time (s)')
    plt.title('Voronoi Closest Zonotope Precomputation Time with %d Zonotopes' %count)

    plt.subplot(212)
    plt.plot(dims,np.log(precomputation_times_avg))
    plt.xlabel('$log$ State Dimension')
    plt.ylabel('$log$ Precomputation Time (s)')
    # plt.title('Voronoi Closest Zonotope Precomputation Time with %d Zonotopes' %count)
    plt.tight_layout()

    if save:
        plt.savefig('precomputation_time' + str(default_timer()) + '.png', dpi=500)

    plt.figure(fig_index)
    fig_index+=1
    plt.subplot(211)
    plt.errorbar(dims,query_times_avg,query_times_std,marker='.',ecolor='r',elinewidth=0.3,capsize=2,linewidth=0.5,markersize=7)
    plt.xlabel('State Dimension')
    plt.ylabel('Query Time (s)')
    plt.title('Voronoi Closest Zonotope Single Query Time with %d Zonotopes' %count)

    plt.subplot(212)
    plt.plot(np.log(dims),np.log(query_times_avg))
    plt.xlabel('$log$ State Dimension')
    plt.ylabel('$log$ Query Time (s)')
    # plt.title('$log$ Voronoi Closest Zonotope Single Query Time with %d Zonotopes' %count)
    plt.tight_layout()

    if save:
        plt.savefig('query_time' + str(default_timer()) + '.png', dpi=500)


    plt.figure(fig_index)
    fig_index+=1
    plt.subplot(211)
    plt.errorbar(dims,query_reduction_percentages_avg,query_reduction_percentages_std,marker='.',ecolor='r',elinewidth=0.3,capsize=2,linewidth=0.5,markersize=7)
    plt.xlabel('State Dimension')
    plt.ylabel('% of Zonotopes Evaluated')
    plt.title('Voronoi Closest Zonotope Reduction Percentage with %d Zonotopes' %count)

    plt.subplot(212)
    plt.plot(np.log(dims),np.log(query_reduction_percentages_avg))
    plt.xlabel('$log$ State Dimension')
    plt.ylabel('$log$ % of Zonotopes Evaluated')
    # plt.title('$log$ Voronoi Closest Zonotope Reduction Percentage with %d Zonotopes' %count)
    plt.tight_layout()

    if save:
        plt.savefig('reduction_percentage' + str(default_timer()) + '.png', dpi=500)
    else:
        plt.show()

if __name__ == '__main__':
    # print('time_against_count(dim=5, counts=np.arange(2, 11, 2) * 50, construction_repeats=3, queries=100)')
    # time_against_count(dim=5, counts=np.arange(2, 11, 2) * 50, construction_repeats=1, queries=100)
    # print('time_against_dim(count=300, dims=np.arange(2, 11, 1), construction_repeats=3, queries=100)')
    # time_against_dim(count = 300, dims=np.arange(2, 7, 2), construction_repeats=1, queries=10, save=False)
    test_voronoi_closest_zonotope(20 , save=False, seed = int(time()))