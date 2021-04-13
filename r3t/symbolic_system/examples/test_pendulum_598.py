import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from r3t.common.help import *

# from pypolycontain.lib.operations import distance_point_polytope_598


reachable_set_epsilon = 2
goal_tolerance = 5e-2
input_limit = 1
input_samples = 9




# assume that we have a class of pendulum that stores the dynamics

# assume Each Node has following structure:
# - Reachable Set: sequence of zonotopes for sliced initial condition
# - Last zonotope for 

# 

def check_collision(parent_node, goal_state, obstacles):
	#assuming parent node has a member called reachable_set that has all the reachable sets

	# assuming obstacles are AABB type
	polytope_list = parent_node.reachable_set

	# what is the interval range that was used to make zonotopes

	# Time T = 30
	polytope_list_sliced = zonotope_slice_34(polytope_list['save_FRS'][0][T][:,1:])  

	#for time T



def generate_keypoints_for_zonotope(Z, k0=[0], kf=[1], N=5):
    ''' slice it for a particular parametere value and then find end point of zonotope.
    keypoints are center of the zonotopes.

    Steps: 

    @ params:
    ---
    Z:      is the last zonotope
    k0:     lowest value of parameter
    kf:     highest value of parameter
    N:      number of keypoints to consider
    '''

    keypoints = []
    slice_dim=[4]
    generator_idx=[]

    ki_list = np.linspace(k0, kf, N)
    # print(ki_list)
    slice_lists = np.vstack(np.meshgrid(*ki_list.T)).reshape(len(k0),-1).T
    # print(slice_lists)

    for slice_value in slice_lists:
        Z_sliced = zonotope_slice(Z, slice_dim=slice_dim, slice_value=slice_value)
        # z, generator_idx=[305, 306, 307], slice_dim=[3, 4, 5], slice_value=[0, 0.2, 0.0004]
        slice_G = Z_sliced.G[slice_dim, slice_idx]






def test_pendulum_planning():
    initial_state = np.zeros(2)

    pendulum_system = Pendulum(initial_state= initial_state, input_limits=np.asarray([[-input_limit],[input_limit]]), m=1, l=0.5, g=9.8, b=0.1)
    
    # We have 2 goals since +180 = -180
    goal_state = np.asarray([np.pi,0.0])
    goal_state_2 = np.asarray([-np.pi,0.0])

    #step size for reachable set
    step_size = 0.3#0.1 # 0.075

    #step size for dynamics propogation
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

    def contains_goal_function(reachable_set, goal_state):
        distance=np.inf

        ## TODO: change contains goal function for including all zonotopes in sequence

        # Check for goal 1: if close nough to parent node, then project to the polytope
        if np.linalg.norm(reachable_set.parent_state-goal_state)<2:
            distance, projection = distance_point_polytope(reachable_set.polytope_list, goal_state)

        # Check for goal 2: if close nough to parent node, then project to the polytope
        elif np.linalg.norm(reachable_set.parent_state-goal_state_2)<2:
            distance, projection = distance_point_polytope(reachable_set.polytope_list, goal_state_2)
        
        # If more than 2 units apart from goal node, then no need to check further.
        else:
            return False, None


        if distance > reachable_set_epsilon:
            return False, None


        #enumerate inputs
        potential_inputs = np.linspace(pendulum_system.input_limits[0,0], pendulum_system.input_limits[1,0], input_samples)
        
        # uses constant control input for propagating dynamics for state
        for u_i in potential_inputs:
            state_list = []
            state=reachable_set.parent_state
            for step in range(int(step_size/nonlinear_dynamic_step_size)):
                state = pendulum_system.forward_step(u=np.atleast_1d(u_i), linearlize=False, modify_system=False, step_size = nonlinear_dynamic_step_size, return_as_env = False,
                     starting_state= state)
                state_list.append(state)
                if np.linalg.norm(goal_state-state)<goal_tolerance:
                    print('Goal error is %d' % np.linalg.norm(goal_state-state))
                    return True, np.asarray(state_list)
                if np.linalg.norm(goal_state_2-state)<goal_tolerance:
                    print('Goal error is %d' % np.linalg.norm(goal_state_2-state))
                    return True, np.asarray(state_list)
        return False, None


    rrt = SymbolicSystem_R3T(pendulum_system, uniform_sampler, step_size, contains_goal_function=contains_goal_function, \
                             use_true_reachable_set=True, use_convex_hull=True)
    found_goal = False
    experiment_name = datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H-%M-%S')

    duration = 0
    os.makedirs('OverR3T_Pendulum_'+experiment_name)
    allocated_time = 2.0# 0.1


    while(1):
        start_time = time.time()
        if rrt.build_tree_to_goal_state(goal_state,stop_on_first_reach=True, allocated_time= allocated_time, rewire=False, explore_deterministic_next_state=False, save_true_dynamics_path=True) is not None:
            found_goal = True
        end_time = time.time()
        #get rrt polytopes
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

        # Plot state tree
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig, ax = visualize_node_tree_2D(rrt, fig, ax, s=0.5, linewidths=0.15, show_path_to_goal=found_goal, goal_override=goal_override)
        # fig, ax = visZ(reachable_polytopes, title="", alpha=0.07, fig=fig,  ax=ax, color='gray')
        # for explored_state in explored_states:
        #     plt.scatter(explored_state[0], explored_state[1], facecolor='red', s=6)
        ax.scatter(initial_state[0], initial_state[1], facecolor='red', s=5)
        ax.scatter(goal_state[0], goal_state[1], facecolor='green', s=5)
        ax.scatter(goal_state[0]-2*np.pi, goal_state[1], facecolor='green', s=5)

        duration += (end_time-start_time)
        plt.title('R3T after %.2f seconds (explored %d nodes)' %(duration, len(polytope_reachable_sets)))
        plt.savefig('R3T_Pendulum_'+experiment_name+'/%.2f_seconds_tree.png' % duration, dpi=500)

        plt.clf()
        plt.close()

        fig, ax = visualize_2D_AH_polytope(reachable_polytopes, fig=fig, ax=ax,N=200,epsilon=0.01, alpha=0.1)

        ax.scatter(initial_state[0], initial_state[1], facecolor='red', s=5)
        ax.scatter(goal_state[0], goal_state[1], facecolor='green', s=5)
        ax.scatter(goal_state[0]-2*np.pi, goal_state[1], facecolor='green', s=5)


if __name__=='__main__':
    # for i in range(1):
    #     test_pendulum_planning()


    mat = scipy.io.loadmat(
        "/home/yingxue/R3T_shared/r3t/overapproximate_with_slice/test_zono.mat"
    )

    Z = 0

    generate_keypoints_for_zonotope(Z)