import pydrake
from r3t.common.r3t_overapprox import *
from polytope_symbolic_system.common.symbolic_system import *
from pypolycontain.lib.operations import distance_point_polytope
from collections import deque
from rtree import index
from closest_polytope_algorithms.bounding_box.polytope_tree import PolytopeTree
from closest_polytope_algorithms.bounding_box.box import AH_polytope_to_box, \
    point_to_box_dmax, point_to_box_distance

class PolytopeReachableSet(ReachableSet):
    def __init__(self, parent_state, polytope_list, generator_idx, k_list, sys=None, epsilon=1e-3, contains_goal_function = None, deterministic_next_state = None, \
                 use_true_reachable_set=False, reachable_set_step_size=None, nonlinear_dynamic_step_size=1e-2):
        ReachableSet.__init__(self, parent_state=parent_state, path_class=PolytopePath)
        self.polytope_list = polytope_list
        self.generator_idx = generator_idx
        self.k_list = k_list
        self.epsilon = epsilon
        self.deterministic_next_state = deterministic_next_state
        self.sys=sys
        self.use_true_reachable_set = use_true_reachable_set
        self.reachable_set_step_size = reachable_set_step_size
        self.nonlinear_dynamic_step_size = nonlinear_dynamic_step_size
        # try:
        #     self.parent_distance = min([distance_point(p, self.parent_state)[0] for p in self.polytope_list])
        # except TypeError:
        #     self.parent_distance = distance_point(self.polytope_list, self.parent_state)[0]

        self.contains_goal_function = contains_goal_function
        # assert(self.parent_distance<self.epsilon)

    def contains(self, goal_state, return_closest_state = True):

        # print(distance_point(self.polytope, goal_state)[0])
        # print(self.polytope)
        try:
            # multimodal
            distance = np.inf
            closest_state = None
            for i, polytope in enumerate(self.polytope_list):
                current_distance, current_closest_state = distance_point_polytope(polytope, goal_state, ball='l2')
                if current_distance < self.epsilon:
                    if return_closest_state:
                        return True, goal_state
                    else:
                        return True
                else:
                    if current_distance < distance:
                        distance = current_distance
                        closest_state = current_closest_state
            return False, closest_state

        except TypeError:

            distance, closest_state = distance_point_polytope(self.polytope_list, goal_state, ball='l2')
            if distance < self.epsilon:
                if return_closest_state:
                    return True, closest_state
                else:
                    return True
            if return_closest_state:
                return False, closest_state
            return False

    def contains_goal(self, goal_state):
        # check if goal is epsilon away from the reachable sets
        if self.contains_goal_function is not None:
            return self.contains_goal_function(self, goal_state)
        raise NotImplementedError


    def find_closest_state_OverR3T(self, query_point, k_closest, K=10):
        # TODO: Add Cost Later
        # u = np.ndarray.flatten(u)[0:self.sys.u.shape[0]]
        # #simulate nonlinear forward dynamics
        state = self.parent_state
        state_list = [self.parent_state]
        for step in range(int(self.reachable_set_step_size/self.nonlinear_dynamic_step_size)):
            u = K * (k_closest - state[1])   # 
            try:
                state = self.sys.forward_step(u=np.atleast_1d(u), linearlize=False, modify_system=False, step_size = self.nonlinear_dynamic_step_size, return_as_env = False,
                        starting_state= state)
                state_list.append(state)
            except Exception as e:
                print('Caught (find_closest_state_OverR3T) %s' %e)
                return np.ndarray.flatten(closest_point), np.asarray([])
        print("initial state:", state_list[0])
        print("final state:", state_list[-1])
        print("k_closest",k_closest)
        # print("state_list", state_list)
        return np.ndarray.flatten(state), state_list


    def plan_collision_free_path_in_set(self, goal_state, return_deterministic_next_state = False):
        try:
            if self.plan_collision_free_path_in_set_function:
                return self.plan_collision_free_path_in_set_function(goal_state, return_deterministic_next_state)
        except AttributeError:
            pass
        #fixme: support collision checking
        #
        # is_contain, closest_state = self.contains(goal_state)
        # if not is_contain:
        #     print('Warning: this should never happen')
        #     return np.linalg.norm(self.parent_state-closest_state), deque([self.parent_state, closest_state]) #FIXME: distance function

        # Simulate forward dynamics if there can only be one state in the next timestep
        if not return_deterministic_next_state:
            return np.linalg.norm(self.parent_state-goal_state), deque([self.parent_state, goal_state])
        return np.linalg.norm(self.parent_state-goal_state), deque([self.parent_state, goal_state]), self.deterministic_next_state

class PolytopePath:
    def __init__(self):
        self.path = deque()
    def __repr__(self):
        return str(self.path)
    def append(self, path):
        self.path+=path #TODO

class PolytopeReachableSetTree(ReachableSetTree):
    '''
    Polytopic reachable set with PolytopeTree
    '''
    def __init__(self, key_vertex_count = 10, distance_scaling_array=None):
        ReachableSetTree.__init__(self)
        self.polytope_tree = None
        self.id_to_reachable_sets = {}
        self.polytope_to_id = {}
        self.key_vertex_count = key_vertex_count
        # for d_neighbor_ids
        # self.state_id_to_state = {}
        # self.state_idx = None
        # self.state_tree_p = index.Property()
        self.distance_scaling_array = distance_scaling_array

    def insert(self, state_id, reachable_set):
        try:
            iter(reachable_set.polytope_list)
            # print("PolytopeReachableSetTree - insert 1 ")
            if self.polytope_tree is None:
                # print("PolytopeReachableSetTree - insert 2 ")

                self.polytope_tree = PolytopeTree(np.array(reachable_set.polytope_list),
                                                  reachable_set.generator_idx,
                                                  reachable_set.k_list,
                                                  key_vertex_count=self.key_vertex_count,
                                                  distance_scaling_array=self.distance_scaling_array)
                # print("PolytopeReachableSetTree - insert 3 ")

            else:
                # print("PolytopeReachableSetTree - insert 4 ")

                self.polytope_tree.insert(np.array(reachable_set.polytope_list), reachable_set.generator_idx, reachable_set.k_list)
                # print("PolytopeReachableSetTree - insert 5 ")

            self.id_to_reachable_sets[state_id] = reachable_set
            # print("PolytopeReachableSetTree - insert 6 ")

            for p in reachable_set.polytope_list:
                # print("PolytopeReachableSetTree - insert 7 ")
                self.polytope_to_id[p] = state_id

        # except TypeError:
        except Exception as e:
            print('Caught (PolytopeReachableSetTree) %s' %e)
            exit()

            # It should NOT go here because we have a zonotope list
            # print("Convex hull, reachable_set.polytope_list is just an AH polytope")
            if self.polytope_tree is None:
                self.polytope_tree = PolytopeTree(np.atleast_1d([reachable_set.polytope_list]).flatten(), reachable_set.generator_idx, reachable_set.k_list, key_vertex_count=self.key_vertex_count,
                                                  distance_scaling_array=self.distance_scaling_array)
                # for d_neighbor_ids
                # self.state_tree_p.dimension = to_AH_polytope(reachable_set.polytope[0]).t.shape[0]
            else:
                self.polytope_tree.insert(np.array([reachable_set.polytope_list]), reachable_set.generator_idx, reachable_set.k_list)
            self.id_to_reachable_sets[state_id] = reachable_set
            self.polytope_to_id[reachable_set.polytope_list] = state_id
        # for d_neighbor_ids
        # state_id = hash(str(reachable_set.parent_state))
        # self.state_idx.insert(state_id, np.repeat(reachable_set.parent_state, 2))
        # self.state_id_to_state[state_id] = reachable_set.parent_state

    def nearest_k_neighbor_ids(self, query_state, k=1, return_state_projection = False):
        # print("nearest_k_neighbor_ids")

        if self.polytope_tree is None:
            return None
        # assert(len(self.polytope_tree.find_closest_polytopes(query_state))==1)
        best_polytope, k_closest, best_distance, state_projection = self.polytope_tree.find_closest_polytopes(query_state, return_state_projection=True, may_return_multiple=True, ball='l2')
        
        # print("len(best_polytope)", len(best_polytope)) # always 1

        best_polytopes_list = [self.polytope_to_id[bp] for bp in best_polytope]

        # print("best_polytopes_list", len(best_polytopes_list))

        if not return_state_projection:
            return best_polytopes_list[:k], k_closest
        return best_polytopes_list[:k], k_closest, best_polytope, [best_distance], [state_projection]

    def d_neighbor_ids(self, query_state, d = np.inf):
        '''

        :param query_state:
        :param d:
        :return:
        '''
        # return self.state_idx.intersection(, objects=False)
        raise NotImplementedError

class SymbolicSystem_StateTree(StateTree):
    def __init__(self, distance_scaling_array=None):
        StateTree.__init__(self)
        self.state_id_to_state = {}
        self.state_tree_p = index.Property()
        self.state_idx = None
        self.distance_scaling_array = distance_scaling_array
    # delayed initialization to consider dimensions
    def initialize(self, dim):
        self.state_tree_p.dimension=dim
        if self.distance_scaling_array is None:
            self.distance_scaling_array = np.ones(dim, dtype='float')
        self.repeated_distance_scaling_array = np.tile(self.distance_scaling_array, 2)
        print('Symbolic System State Tree dimension is %d-D' % self.state_tree_p.dimension)
        self.state_idx = index.Index(properties=self.state_tree_p)

    def insert(self, state_id, state):
        if not self.state_idx:
            self.initialize(state.shape[0])
        scaled_state = np.multiply(self.distance_scaling_array,
                                   state)
        self.state_idx.insert(state_id, np.tile(scaled_state, 2))
        self.state_id_to_state[state_id] = state

    def state_ids_in_reachable_set(self, query_reachable_set):
        assert(self.state_idx is not None)
        try:
            state_ids_list = []
            for p in query_reachable_set.polytope_list:
                lu = AH_polytope_to_box(p)
                scaled_lu = np.multiply(self.repeated_distance_scaling_array,lu)
                state_ids_list.extend(list(self.state_idx.intersection(scaled_lu)))
            return state_ids_list
        except TypeError:
            lu = AH_polytope_to_box(query_reachable_set.polytope_list)
            scaled_lu = np.multiply(self.repeated_distance_scaling_array, lu)
            return list(self.state_idx.intersection(scaled_lu))

class SymbolicSystem_OverR3T(OverR3T):
    def __init__(self, sys, sampler, step_size, contains_goal_function = None, compute_last_reachable_set=None, use_true_reachable_set=False, \
                 nonlinear_dynamic_step_size=1e-2, use_convex_hull=True, goal_tolerance = 1e-2):
        self.sys = sys
        self.step_size = step_size
        self.contains_goal_function = contains_goal_function
        self.goal_tolerance = goal_tolerance
        if compute_last_reachable_set is None:
            def compute_last_reachable_set(state, reachable_set_polytope, generator_idx, k_list):
                '''
                Compute polytopic reachable set using the system - for last time index
                :param h:
                :return:
                '''
                deterministic_next_state = None
                # print("compute_last_reachable_set X ",reachable_set_polytope.x.shape)

                # if np.all(self.sys.get_linearization(state=state).B == 0):
                #     if use_true_reachable_set:
                #         deterministic_next_state=[state]
                #         for step in range(int(self.step_size / nonlinear_dynamic_step_size)):
                #             state = self.sys.forward_step(starting_state=state, modify_system=False, return_as_env=False, step_size=nonlinear_dynamic_step_size)
                #             deterministic_next_state.append(state)
                #     else:
                #         deterministic_next_state = [state, self.sys.forward_step(starting_state=state, modify_system=False, return_as_env=False, step_size=self.step_size)]
                return PolytopeReachableSet(state, reachable_set_polytope, generator_idx, k_list, sys=self.sys, contains_goal_function=self.contains_goal_function, \
                                            deterministic_next_state=deterministic_next_state, reachable_set_step_size=self.step_size, use_true_reachable_set=use_true_reachable_set,\
                                            nonlinear_dynamic_step_size=nonlinear_dynamic_step_size)
        OverR3T.__init__(self, self.sys.get_current_state(), compute_last_reachable_set, sampler, PolytopeReachableSetTree, SymbolicSystem_StateTree, PolytopePath)