import numpy as np
import os
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import matplotlib
from visualization.visualize import *

matplotlib.rcParams['font.family'] = "Times New Roman"
matplotlib.rcParams.update({'font.size': 15})

# Load
hopper_dir = '/Users/albertwu/Google Drive/MIT/RobotLocomotion/Closest Polytope/ACC2020/Results/rrt_hopper/test_on_rrt20190925_18-00-43'#os.path.dirname(os.path.realpath(__file__))+'/../tests/test_random_zonotope_dim20190919_01-56-56'
pendulum_dir = '/Users/albertwu/Google Drive/MIT/RobotLocomotion/Closest Polytope/ACC2020/Results/rrt_pendulum/test_on_rrt20190925_16-23-31'#os.path.dirname(os.path.realpath(__file__))+'/../tests/test_random_zonotope_dim20190919_01-56-56'
mpc_rod_dir = '/Users/albertwu/Google Drive/MIT/RobotLocomotion/Closest Polytope/ACC2020/Results/mpc_manipulation/test_on_mpc20190926_17-19-52'#os.path.dirname(os.path.realpath(__file__))+'/../tests/test_random_zonotope_dim20190919_01-56-56'
mpc_pendulum_dir = '/Users/albertwu/Google Drive/MIT/RobotLocomotion/Closest Polytope/ACC2020/Results/mpc_pendulum/data'#os.path.dirname(os.path.realpath(__file__))+'/../tests/test_random_zonotope_dim20190919_01-56-56'


hopper_aabb_query_times_median = np.load(hopper_dir+'/aabb_query_times_median.npy')
hopper_aabb_query_times_max = np.load(hopper_dir+'/aabb_query_times_max.npy')
hopper_aabb_query_reduction_percentages_median = np.load(hopper_dir+'/aabb_query_reduction_percentages_median.npy')
hopper_aabb_query_reduction_percentages_max = np.load(hopper_dir+'/aabb_query_reduction_percentages_max.npy')

pendulum_aabb_query_times_median = np.load(pendulum_dir+'/aabb_query_times_median.npy')
pendulum_aabb_query_times_max = np.load(pendulum_dir+'/aabb_query_times_max.npy')
pendulum_aabb_query_reduction_percentages_median = np.load(pendulum_dir+'/aabb_query_reduction_percentages_median.npy')
pendulum_aabb_query_reduction_percentages_max = np.load(pendulum_dir+'/aabb_query_reduction_percentages_max.npy')


mpc_rod_aabb_query_times_median = np.load(mpc_rod_dir+'/aabb_query_times_median.npy')
mpc_rod_aabb_query_times_max = np.load(mpc_rod_dir+'/aabb_query_times_max.npy')
mpc_rod_aabb_query_reduction_percentages_median = np.load(mpc_rod_dir+'/aabb_query_reduction_percentages_median.npy')
mpc_rod_aabb_query_reduction_percentages_max = np.load(mpc_rod_dir+'/aabb_query_reduction_percentages_max.npy')

mpc_pendulum_aabb_query_times_median = np.load(mpc_pendulum_dir+'/aabb_query_times_median.npy')
mpc_pendulum_aabb_query_times_max = np.load(mpc_pendulum_dir+'/aabb_query_times_max.npy')
mpc_pendulum_aabb_query_reduction_percentages_median = np.load(mpc_pendulum_dir+'/aabb_query_reduction_percentages_median.npy')
mpc_pendulum_aabb_query_reduction_percentages_max = np.load(mpc_pendulum_dir+'/aabb_query_reduction_percentages_max.npy')



hopper_polytope_counts = np.load(hopper_dir+'/polytope_counts.npy')
pendulum_polytope_counts = np.asarray([159,357,678,1099,1657,2518,3679,5308,7387,10106])
mpc_rod_polytope_counts = np.load(mpc_rod_dir+'/polytope_counts.npy')
mpc_pendulum_polytope_counts = np.load(mpc_pendulum_dir+'/polytope_counts.npy')


hopper_aabb_query_count_median = np.multiply(hopper_polytope_counts, hopper_aabb_query_reduction_percentages_median)/100
hopper_aabb_query_count_max = np.multiply(hopper_polytope_counts, hopper_aabb_query_reduction_percentages_max)/100
mpc_rod_aabb_query_count_median = np.multiply(mpc_rod_polytope_counts, mpc_rod_aabb_query_reduction_percentages_median)/100
mpc_rod_aabb_query_count_max = np.multiply(mpc_rod_polytope_counts, mpc_rod_aabb_query_reduction_percentages_max)/100


pendulum_aabb_query_count_median = np.multiply(pendulum_polytope_counts, pendulum_aabb_query_reduction_percentages_median)/100
pendulum_aabb_query_count_max = np.multiply(pendulum_polytope_counts, pendulum_aabb_query_reduction_percentages_max)/100
mpc_pendulum_aabb_query_count_median = np.multiply(mpc_pendulum_polytope_counts, mpc_pendulum_aabb_query_reduction_percentages_median)/100
mpc_pendulum_aabb_query_count_max = np.multiply(mpc_pendulum_polytope_counts, mpc_pendulum_aabb_query_reduction_percentages_max)/100

# Plot
fig_index = 0
# plt.figure(fig_index)
# fig_index += 1
# plt.subplot(111)
# plt.plot(polytope_counts, voronoi_precomputation_times_median, marker='.', color='b',
#             linewidth=0.5, markersize=7)
# plt.plot(polytope_counts, aabb_precomputation_times_median, marker='.', color='r', linewidth=0.5, markersize=7)
# plt.legend(['Triangle Ineq.', 'AABB'])
# plt.xlabel('Number of Polytopes')
# plt.ylabel('Precomputation Time (s)')
# plt.title('Precomputation Time vs. Number of Polytopes')
#
# plt.subplot(212)
# plt.plot(dims, np.log(voronoi_precomputation_times_median))
# plt.xlabel('$log$ State Dimension')
# plt.ylabel('$log$ Precomputation Time (s)')
# # plt.title('Voronoi Closest Zonotope Precomputation Time with %d Zonotopes' %count)
plt.tight_layout()

plt.figure(fig_index)
fig_index += 1
plt.subplot(111)

# plt.plot(hopper_polytope_counts, hopper_aabb_query_times_median, marker='.', color='b', linewidth=0.5, markersize=7)
# plt.plot(pendulum_polytope_counts, pendulum_aabb_query_times_median, marker='.', color='r', linewidth=0.5, markersize=7)
# plt.legend(['Hopper', 'Pendulum'])
# plt.xlabel('Number of Polytopes')
# plt.ylabel('Query Time (s)')
# # plt.yscale('log')
# # plt.xscale('log')
# # plt.ylim(bottom=0 , top=0.02)
# plt.title('AABB Median Query Time on R3T Datasets')
# plt.tight_layout()
#
# plt.savefig('query_time_median.png', dpi=500)
# plt.close()
percent_x = np.linspace(0, 11000, 50)
five_percent_line = 0.05*percent_x
ten_percent_line = 0.1*percent_x
x_eq_y = percent_x
# plt.plot(hopper_polytope_counts, hopper_aabb_query_count_median, marker='.', color='b', linewidth=0.5, markersize=7)
# plt.plot(pendulum_polytope_counts, pendulum_aabb_query_count_median, marker='.', color='r', linewidth=0.5, markersize=7)
plt.plot(hopper_polytope_counts, hopper_aabb_query_count_median, marker='.', color='b', linewidth=0.5, markersize=7)
plt.plot(pendulum_polytope_counts, pendulum_aabb_query_count_median, marker='.', color='r', linewidth=0.5, markersize=7)
plt.plot(mpc_rod_polytope_counts, mpc_rod_aabb_query_count_median, marker='.', color='g', linewidth=0.5, markersize=7)
plt.plot(mpc_pendulum_polytope_counts, mpc_pendulum_aabb_query_count_median, marker='.', color='m', linewidth=0.5, markersize=7)

# plt.plot(percent_x, five_percent_line, '-', color='k', alpha=0.2)
# plt.plot(percent_x, ten_percent_line, '-', color='k', alpha=0.8)
plt.legend(['R3T Hopper', 'R3T Pendulum', 'MPC Rod', 'MPC Pendulum'])
plt.xlabel('Number of Polytopes')
plt.ylabel('Number of Polytopes Evaluated')
# plt.yscale('log')
plt.xscale('log')
# plt.ylim(bottom=0, top=0.02)
plt.title('AABB Median Evaluated Polytopes on R3T and MPC Datasets')
plt.tight_layout()

plt.savefig('count_median.png', dpi=500)
plt.close()

# plt.plot(polytope_counts, np.multiply(aabb_query_reduction_percentages_median/100, polytope_counts), marker='.', color='r',
#              linewidth=0.5, markersize=7)
# # plt.legend(['Triangle Ineq.', 'AABB'])
# plt.xlabel('Number of Polytopes')
# plt.ylabel('Query Time (s)')
# # plt.yscale('log')
# plt.xscale('log')
# # plt.ylim(bottom=0, top=0.02)
# plt.title('AABB Max Query Time on R3T Hopper')
# print(hopper_polytope_counts, hopper_aabb_query_times_max, pendulum_aabb_query_times_max, pendulum_polytope_counts)
plt.plot(hopper_polytope_counts, hopper_aabb_query_count_max, marker='.', color='b', linewidth=0.5, markersize=7)
plt.plot(pendulum_polytope_counts, pendulum_aabb_query_count_max, marker='.', color='r', linewidth=0.5, markersize=7)
plt.plot(mpc_rod_polytope_counts, mpc_rod_aabb_query_count_max, marker='.', color='g', linewidth=0.5, markersize=7)
plt.plot(mpc_pendulum_polytope_counts, mpc_pendulum_aabb_query_count_max, marker='.', color='m', linewidth=0.5, markersize=7)
plt.plot(percent_x, x_eq_y, 'k--', alpha=0.3, linewidth=0.7)
plt.legend(['R3T Hopper', 'R3T Pendulum', 'MPC Rod', 'MPC Pendulum', '$x=y$'])
plt.xlabel('Number of Polytopes')
plt.ylabel('Number of Polytopes Evaluated')
# plt.yscale('log')
# plt.xscale('log')
plt.ylim(bottom=-100, top=3250)
plt.title('AABB Max Evaluated Polytopes on R3T and MPC Datasets')

plt.tight_layout()
plt.savefig('count_max.png', dpi=500)
# plt.subplot(212)
# plt.plot(np.log(dims), np.log(voronoi_query_times_median))
# plt.xlabel('$log$ State Dimension')
# plt.ylabel('$log$ Query Time (s)')
# # plt.title('$log$ Voronoi Closest Zonotope Single Query Time with %d Zonotopes' %count)