import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from timeit import default_timer
from polytope_symbolic_system.examples.pendulum import Pendulum
from r3t.symbolic_system.symbolic_system_r3t_overapprox import SymbolicSystem_OverR3T
from pypolycontain.visualization.visualize_2D import visualize_2D_AH_polytope, visualize_obs
from pypolycontain.lib.operations import distance_point_polytope, to_AH_polytope
from pypolycontain.utils.random_polytope_generator import get_k_random_edge_points_in_zonotope_OverR3T
from r3t.utils.visualization import visualize_node_tree_2D
import time
from datetime import datetime
import os
import scipy.io
from r3t.common.help import *

# import warnings
# warnings.filterwarnings("ignore")

matplotlib.rcParams['font.family'] = "Times New Roman"
matplotlib.rcParams.update({'font.size': 14})

reachable_set_epsilon = 2
goal_tolerance = 1. # 0.5 # 5e-2
input_limit = 200
input_samples = 9


def test_pendulum_planning():
    initial_state = np.zeros(2)

    pendulum_system = Pendulum(initial_state= initial_state, input_limits=np.asarray([[-input_limit],[input_limit]]), m=1, l=0.5, g=9.81, b=0.1)
    goal_state = np.asarray([np.pi,0.0])
    goal_state_2 = np.asarray([-np.pi,0.0])
    step_size = 0.3     #0.1 # 0.075
    nonlinear_dynamic_step_size=1e-2
    def uniform_sampler():
        rnd = np.random.rand(2)
        rnd[0] = (rnd[0]-0.5)*2*1.5*np.pi
        rnd[1] = (rnd[1]-0.5)*2*12
        goal_bias_rnd = np.random.rand(1)
        # if goal_bias_rnd <0.2:
        #     return goal_state + [2*np.pi*np.random.randint(-1,1),0] + [np.random.normal(0,0.8),np.random.normal(0,1.5)]
        return rnd

    def gaussian_mixture_sampler():
        gaussian_ratio = 0.0
        rnd = np.random.rand(2)
        rnd[0] = np.random.normal(goal_state[0],1)
        rnd[1] = np.random.normal(goal_state[1],1)
        if np.random.rand(1) > gaussian_ratio:
            return uniform_sampler()
        return rnd

    def ring_sampler():
        theta = np.random.rand(1)*2*np.pi
        rnd = np.zeros(2)
        r = np.random.rand(1)+2.5
        rnd[0] = r*np.cos(theta)
        rnd[1] = r*np.sin(theta)
        return rnd

    def contains_goal_function(reachable_set, goal_state, key_vertex_count=3):

        distance = np.inf
        # distance1 = np.inf
        # distance2 = np.inf
        if np.linalg.norm(reachable_set.parent_state-goal_state)<2:
            goal = goal_state
            for P in reachable_set.polytope_list:
                P_projected = project_zonotope(P, dim=[0, 1], mode='full')
                distance, projection = distance_point_polytope(P_projected, goal_state)
                # distance1 = min(distance1, distance)
        elif np.linalg.norm(reachable_set.parent_state-goal_state_2)<2:
            goal = goal_state_2
            for P in reachable_set.polytope_list:
                P_projected = project_zonotope(P, dim=[0, 1], mode='full')
                distance, projection = distance_point_polytope(P_projected, goal_state_2)
                # distance2 = min(distance2, distance)
        else:
            return False, None

        if distance > reachable_set_epsilon:
            return False, None

        # if min(distance1, distance2) > reachable_set_epsilon:
        #     return False, None
        # if distance1 < distance2:
        #     goal = goal_state
        # else:
        #     goal = goal_state_2

        # slice polyoptes to get keypoint
        goal_key_points = np.zeros((len(reachable_set.polytope_list)*key_vertex_count, 2))
        keypoint_k_lists = np.zeros((len(reachable_set.polytope_list)*key_vertex_count, 1))
            
        for i, k in enumerate(reachable_set.k_list):
            p = reachable_set.polytope_list[i]
            i_key_points, keypoint_k_list = get_k_random_edge_points_in_zonotope_OverR3T(p, reachable_set.generator_idx[i], N=key_vertex_count, k=k, lk=-0.5, uk=0.5) 
            goal_key_points[i*key_vertex_count:(i+1)*key_vertex_count, :] = i_key_points
            keypoint_k_lists[i*key_vertex_count:(i+1)*key_vertex_count, :] = keypoint_k_list

        # Find closest keypoint
        delta = goal_key_points - goal

        # ball = "l2"
        # if ball=="infinity":
        #     d=np.linalg.norm(delta,ord=np.inf, axis=1)
        # elif ball=="l1":
        #     d=np.linalg.norm(delta,ord=1, axis=1)
        # elif ball=="l2":
        #     d=np.linalg.norm(delta, ord=2, axis=1)
        # else:
        #     raise NotImplementedError
        d=np.linalg.norm(delta, ord=2, axis=1)
            
        # get the parameter of closest_keypoint
        k_closest = keypoint_k_lists[np.argmin(d), :]

        # propagate
        state = reachable_set.parent_state
        state_list = [reachable_set.parent_state]
        for step in range(int(reachable_set.reachable_set_step_size/reachable_set.nonlinear_dynamic_step_size)):
            K = 2
            u = K * k_closest
            # u = K * (k_closest - state[0]) # theta
            # u = K * (k_closest - state[1]) # theta_dot
            state = reachable_set.sys.forward_step(u=np.atleast_1d(u), linearlize=False, modify_system=False, step_size = reachable_set.nonlinear_dynamic_step_size, return_as_env = False,
                    starting_state=state)
            state_list.append(state)

        delta = state-goal

        # if ball=="infinity":
        #     d=np.linalg.norm(delta,ord=np.inf)
        # elif ball=="l1":
        #     d=np.linalg.norm(delta,ord=1)
        # elif ball=="l2":
        #     d=np.linalg.norm(delta, ord=2)
        # else:
        #     raise NotImplementedError
        d=np.linalg.norm(delta, ord=2)

        print("closest distance of keypoint to goal: ", d)

        if d<goal_tolerance:
            return True, state_list
        else:
            return False, None


    rrt = SymbolicSystem_OverR3T(pendulum_system, uniform_sampler, step_size, contains_goal_function=contains_goal_function, \
                             use_true_reachable_set=True, use_convex_hull=True)
    found_goal = False
    experiment_name = datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H-%M-%S')

    duration = 0
    os.makedirs('OverR3T_Pendulum_'+experiment_name)
    allocated_time = 10.0 # 0.1

    VISUALIZE = True
    iter_count = 0

    # Define obstacles
    Z_obs_list = []
    G_l = np.array([[0.1, 0], [0, 0.3]])*1.5
    x_l = np.array([1., 4.5]).reshape(2, 1)
    z_obs = zonotope(x_l, G_l)
    Z_obs_list.append(z_obs)

    while(1):
        print("iter_count: ", iter_count)
        iter_count += 1

        start_time = time.time()
        if rrt.build_tree_to_goal_state(goal_state, Z_obs_list=Z_obs_list, stop_on_first_reach=True, allocated_time=allocated_time, rewire=False, explore_deterministic_next_state=False, save_true_dynamics_path=True) is not None:
            found_goal = True
        end_time = time.time()
        # get rrt polytopes
        polytope_reachable_sets = rrt.reachable_set_tree.id_to_reachable_sets.values()
        reachable_polytopes = []
        explored_states = []
        for prs in polytope_reachable_sets:
            reachable_polytopes.append(prs.polytope_list)
            explored_states.append(prs.parent_state)

        goal_override = None
        if found_goal:
            p = rrt.goal_node.parent.state
            if np.linalg.norm(p-np.asarray([np.pi,0.0])) < np.linalg.norm(p-np.asarray([-np.pi,0.0])):
                goal_override = np.asarray([np.pi,0.0])
            else:
                goal_override = np.asarray([-np.pi, 0.0])

        print("Number of Nodes in the tree: ", rrt.node_tally)

        if VISUALIZE:
            # Plot state tree
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig, ax = visualize_node_tree_2D(rrt, fig, ax, s=2, linewidths=0.25, show_path_to_goal=found_goal, goal_override=goal_override)
            # fig, ax = visZ(reachable_polytopes, title="", alpha=0.07, fig=fig,  ax=ax, color='gray')
            # for explored_state in explored_states:
            #     plt.scatter(explored_state[0], explored_state[1], facecolor='red', s=6)
            ax.scatter(initial_state[0], initial_state[1], facecolor='red', s=5)
            ax.scatter(goal_state[0], goal_state[1], facecolor='green', s=5)
            ax.scatter(goal_state[0]-2*np.pi, goal_state[1], facecolor='green', s=5)

            # visualize obstacles
            fig, ax = visualize_obs(Z_obs_list, color='red', fig=fig, ax=ax, N=50, epsilon=0.01, alpha=0.8) 

            # ax.grid(True, which='both')
            # y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
            # ax.yaxis.set_major_formatter(y_formatter)
            # ax.set_yticks(np.arange(-10, 10, 2))
            # ax.set_xlim([-4, 4])
            # ax.set_ylim([-10, 10])
            # ax.set_xlabel('$\\theta (rad)$')
            # ax.set_ylabel('$\dot{\\theta} (rad/s)$')
            
            duration += (end_time-start_time)
            # plt.title('R3T after %.2f seconds (explored %d nodes)' %(duration, len(polytope_reachable_sets)))
            # plt.savefig('R3T_Pendulum_'+experiment_name+'/%.2f_seconds_tree.png' % duration, dpi=500)
            # # plt.show()
            # plt.xlim([-4, 4])
            # plt.ylim([-10,10])
            # plt.clf()
            # plt.close()

            # Plot explored reachable sets
            # FIXME: Handle degenerated reachable set
            # fig = plt.figure()
            # ax = fig.add_subplot(111)

            fig, ax = visualize_2D_AH_polytope(reachable_polytopes, states=explored_states, fig=fig, ax=ax,N=50,epsilon=0.01, alpha=0.1)

            ax.scatter(initial_state[0], initial_state[1], facecolor='red', s=5)
            ax.scatter(goal_state[0], goal_state[1], facecolor='green', s=5)
            ax.scatter(goal_state[0]-2*np.pi, goal_state[1], facecolor='green', s=5)

            # ax.set_aspect('equal')
            plt.xlabel('$x$')
            plt.ylabel('$\dot{x}$')
            plt.xlim([-5, 5])
            plt.ylim([-12,12])
            plt.tight_layout()
            plt.title('$|u| \leq %.2f$ Reachable Set after %.2fs (%d nodes)' %(input_limit, duration, len(polytope_reachable_sets)))
            plt.savefig('OverR3T_Pendulum_'+experiment_name+'/%.2f_seconds_reachable_sets.png' % duration, dpi=500)
            # plt.show()
            # plt.pause(0.2)
            plt.clf()
            plt.close()
            #
            # if found_goal:
            #     break
            # allocated_time*=5

if __name__=='__main__':

    for i in range(1):
        test_pendulum_planning()