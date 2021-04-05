import numpy as np
import os
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import matplotlib
from visualization.visualize import *

matplotlib.rcParams['font.family'] = "Times New Roman"
matplotlib.rcParams.update({'font.size': 15})

# Load
# Uniform
# dir = '/Users/albertwu/Google Drive/MIT/RobotLocomotion/Closest Polytope/ACC2020/Results/uniform_dim/test_random_zonotope_dim20190925_20-25-57'
# Line
dir = '/Users/albertwu/Google Drive/MIT/RobotLocomotion/Closest Polytope/ACC2020/Results/line_dim/test_random_zonotope_dim20190926_10-10-31'


voronoi_precomputation_times_median = np.load(dir+'/voronoi_precomputation_times_median.npy')
voronoi_precomputation_times_min = np.load(dir+'/voronoi_precomputation_times_min.npy')
voronoi_precomputation_times_max = np.load(dir+'/voronoi_precomputation_times_max.npy')
voronoi_query_times_median = np.load(dir+'/voronoi_query_times_median.npy')
voronoi_query_times_min = np.load(dir+'/voronoi_query_times_min.npy')
voronoi_query_times_max = np.load(dir+'/voronoi_query_times_max.npy')
voronoi_query_reduction_percentages_median = np.load(dir+'/voronoi_query_reduction_percentages_median.npy')
voronoi_query_reduction_percentages_min = np.load(dir+'/voronoi_query_reduction_percentages_min.npy')
voronoi_query_reduction_percentages_max = np.load(dir+'/voronoi_query_reduction_percentages_max.npy')

aabb_precomputation_times_median = np.load(dir+'/aabb_precomputation_times_median.npy')
aabb_precomputation_times_min = np.load(dir+'/aabb_precomputation_times_min.npy')
aabb_precomputation_times_max = np.load(dir+'/aabb_precomputation_times_max.npy')
aabb_query_times_median = np.load(dir+'/aabb_query_times_median.npy')
aabb_query_times_min = np.load(dir+'/aabb_query_times_min.npy')
aabb_query_times_max = np.load(dir+'/aabb_query_times_max.npy')
aabb_query_reduction_percentages_median = np.load(dir+'/aabb_query_reduction_percentages_median.npy')
aabb_query_reduction_percentages_min = np.load(dir+'/aabb_query_reduction_percentages_min.npy')
aabb_query_reduction_percentages_max = np.load(dir+'/aabb_query_reduction_percentages_max.npy')


params = np.load(dir+'/params.npy')
print(params)

for p in params:
    if p[0] == 'count':
        counts = p[1]
    if p[0] == 'dim':
        dims = p[1]

voronoi_query_count_median = np.multiply(dims, voronoi_query_reduction_percentages_median)/100
voronoi_query_count_min =np.multiply(dims, voronoi_query_reduction_percentages_min)/100
voronoi_query_count_max =np.multiply(dims, voronoi_query_reduction_percentages_max)/100

aabb_query_count_median = np.multiply(dims, aabb_query_reduction_percentages_median)/100
aabb_query_count_min = np.multiply(dims, aabb_query_reduction_percentages_min)/100
aabb_query_count_max = np.multiply(dims, aabb_query_reduction_percentages_max)/100

# Plot
fig_index = 0
plt.figure(fig_index)
fig_index += 1
plt.subplot(111)

f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

line2 =ax2.errorbar(dims, aabb_precomputation_times_median, np.vstack([aabb_precomputation_times_median-aabb_precomputation_times_min, aabb_precomputation_times_max-aabb_precomputation_times_median]), marker='.', color='r', ecolor='r', elinewidth=0.3,
             capsize=2,
             linewidth=0.5, markersize=7)
line =ax.errorbar(dims, voronoi_precomputation_times_median, np.vstack([voronoi_precomputation_times_median-voronoi_precomputation_times_min, voronoi_precomputation_times_max-voronoi_precomputation_times_median]), marker='.', color='b', ecolor='b', elinewidth=0.3,
             capsize=2,
             linewidth=0.5, markersize=7)
ax.set_ylim([80,150])
ax2.set_ylim([0,5])
f.legend((line, line2), ('Triangle Ineq.','AABB'), loc='center')

ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop='off')  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

plt.xlabel('Dimension')
f.text(0.02,0.5, "Precomputation Time (s)", ha="center", va="center", rotation=90)
f.text(0.5,0.93, "Dimension Scalability with Linearly Distributed Polytopes", ha="center", va="center")

# plt.tight_layout()
plt.savefig('precompute_line_dim_time.png', dpi=500)
# plt.subplot(212)
# plt.plot(dims, np.log(voronoi_precomputation_times_median))
# plt.xlabel('$log$ State Dimension')
# plt.ylabel('$log$ Precomputation Time (s)')
# # plt.title('Voronoi Closest Zonotope Precomputation Time with %d Zonotopes' %count)
plt.tight_layout()
plt.close()

plt.figure(fig_index)
fig_index += 1
plt.subplot(111)
plt.errorbar(dims, voronoi_query_times_median, np.vstack([voronoi_query_times_median-voronoi_query_times_min, voronoi_query_times_max-voronoi_query_times_median]), marker='.', color='b', ecolor='b', elinewidth=0.3,
             capsize=2,
             linewidth=0.5, markersize=7)
plt.errorbar(dims, aabb_query_times_median, np.vstack([aabb_query_times_median-aabb_query_times_min, aabb_query_times_max-aabb_query_times_median]), marker='.', color='r', ecolor='r', elinewidth=0.3,
             capsize=2,
             linewidth=0.5, markersize=7)
plt.legend(['Triangle Ineq.', 'AABB'])
plt.xlabel('Dimension')
plt.ylabel('Query Time (s)')
# plt.yscale('log')
plt.title('Dimension Scalability with Linearly Distributed Polytopes')

plt.tight_layout()
plt.savefig('online_line_dim_time.png', dpi=500)
# plt.subplot(212)
# plt.plot(np.log(dims), np.log(voronoi_query_times_median))
# plt.xlabel('$log$ State Dimension')
# plt.ylabel('$log$ Query Time (s)')
# # plt.title('$log$ Voronoi Closest Zonotope Single Query Time with %d Zonotopes' %count)
plt.close()

plt.figure(fig_index)
fig_index += 1
plt.subplot(111)
plt.errorbar(dims, voronoi_query_reduction_percentages_median, np.vstack([voronoi_query_reduction_percentages_median-voronoi_query_reduction_percentages_min, voronoi_query_reduction_percentages_max-voronoi_query_reduction_percentages_median]), marker='.',
             color='b', ecolor='b',
             elinewidth=0.3, capsize=2, linewidth=0.5, markersize=7)
plt.errorbar(dims, aabb_query_reduction_percentages_median, np.vstack([aabb_query_reduction_percentages_median-aabb_query_reduction_percentages_min, aabb_query_reduction_percentages_max-aabb_query_reduction_percentages_median]), marker='.', color='r',
             ecolor='r',
             elinewidth=0.3, capsize=2, linewidth=0.5, markersize=7)
plt.legend(['Triangle Ineq.', 'AABB'])
plt.xlabel('Dimension')
plt.ylabel('% of Polytopes Evaluated')
plt.title('Dimension Scalability with Linearly Distributed Polytopes')

# plt.title('Nearest Polytope Evaluated Percentage vs. Number of Polytopes')
plt.tight_layout()
plt.savefig('online_line_dim_reduction.png', dpi=500)

plt.figure(fig_index)
fig_index += 1
plt.subplot(111)
plt.errorbar(dims, counts / 100 * voronoi_query_reduction_percentages_median, counts / 100 * np.vstack(
    [voronoi_query_reduction_percentages_median - voronoi_query_reduction_percentages_min,
     voronoi_query_reduction_percentages_max - voronoi_query_reduction_percentages_median]), marker='.', color='b',
             ecolor='b',
             elinewidth=0.3, capsize=2, linewidth=0.5, markersize=7)
plt.errorbar(dims, counts / 100 * aabb_query_reduction_percentages_median,
             counts / 100 * np.vstack([aabb_query_reduction_percentages_median - aabb_query_reduction_percentages_min,
                                      aabb_query_reduction_percentages_max - aabb_query_reduction_percentages_median]),
             marker='.', color='r', ecolor='r',
             elinewidth=0.3, capsize=2, linewidth=0.5, markersize=7)
plt.legend(['Triangle Ineq.', 'AABB'])
plt.xlabel('Dimension')
plt.ylabel('Number of Polytopes Evaluated')
plt.title('Dimension Scalability with Linearly Distributed Polytopes')

# plt.title('Nearest Polytope Evaluated Percentage vs. Number of Polytopes')
plt.tight_layout()
plt.savefig('online_line_dim_count.png', dpi=500)


# plt.subplot(212)
# plt.plot(np.log(dims), np.log(voronoi_query_reduction_percentages_median))
# plt.xlabel('$log$ State Dimension')
# plt.ylabel('$log$ % of Zonotopes Evaluated')
# # plt.title('$log$ Voronoi Closest Zonotope Reduction Percentage with %d Zonotopes' %count)
# plt.show()