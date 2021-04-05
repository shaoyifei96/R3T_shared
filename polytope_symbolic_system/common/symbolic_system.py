import pydrake.symbolic as sym
import numpy as np
from pypolycontain.lib.zonotope import *
from pypolycontain.lib.operations import convex_hull_of_point_and_polytope

def extract_variable_value_from_env(symbolic_var, env):
    # symbolic_var is a vector
    var_value = np.zeros(symbolic_var.shape[0])
    for i in range(symbolic_var.shape[0]):
        var_value[i] = env[symbolic_var[i]]
    return var_value

class Dynamics:
    def __init__(self):
        self.type='undefined'
        pass

    def construct_linearized_system_at(self, env):
        raise NotImplementedError


class ContinuousLinearDynamics(Dynamics):
    def __init__(self, A, B, c):
        Dynamics.__init__(self)
        self.A = A
        self.B = B
        self.c = c
        self.type='continuous'

    def construct_linearized_system_at(self, env):
        print('Warning: system is already linear!')
        return self

    def evaluate_xdot(self, x, u):
        return np.dot(self.A,x)+np.dot(self.B,u)+self.c


class ContinuousDynamics(Dynamics):
    '''
    System described by xdot(t) = f(x(t), u(t))
    '''
    def __init__(self, f, x, u):
        Dynamics.__init__(self)
        self.f = f
        self.x = x
        self.u = u
        self.type='continuous'
        self._linearlize_system()

    def _linearlize_system(self):
        self.A = sym.Jacobian(self.f, self.x)
        self.B = sym.Jacobian(self.f, self.u)
        self.c = -(np.dot(self.A, self.x)+np.dot(self.B, self.u))+self.f

    def construct_linearized_system_at(self, env):
        return ContinuousLinearDynamics(sym.Evaluate(self.A, env), sym.Evaluate(self.B, env), sym.Evaluate(self.c, env))

    def evaluate_xdot(self, env, linearize):
        if linearize:
            linsys = self.construct_linearized_system_at(env)
            x_env = extract_variable_value_from_env(self.x, env)
            u_env = extract_variable_value_from_env(self.u, env)
            return linsys.evaluate_xdot(x_env, u_env)
        else:
            return sym.Evaluate(self.f, env)

class DiscreteLinearDynamics(Dynamics):
    def __init__(self, A, B, c):
        Dynamics.__init__(self)
        self.A = A
        self.B = B
        self.c = c
        self.type='discrete'

    def construct_linearized_system_at(self, env):
        print('Warning: system is already linear!')
        return self

    def evaluate_x_next(self, x, u):
        return np.dot(self.A,x)+np.dot(self.B,u)+self.c

class DiscreteDynamics(Dynamics):
    '''
    System described by x[t+1] = f(x[t], u[t])
    '''
    def __init__(self, f, x, u):
        Dynamics.__init__(self)
        self.f = f
        self.x = x
        self.u = u
        self.type='discrete'
        self._linearlize_system()

    def _linearlize_system(self):
        self.A = sym.Jacobian(self.f, self.x)
        self.B = sym.Jacobian(self.f, self.u)
        self.c = -(np.dot(self.A, self.x)+np.dot(self.B, self.u))+self.f

    def construct_linearized_system_at(self, env):
        return DiscreteLinearDynamics(sym.Evaluate(self.A, env), sym.Evaluate(self.B, env), sym.Evaluate(self.c, env))

    def evaluate_x_next(self, env, linearize=False):
        if linearize:
            linsys = self.construct_linearized_system_at(env)
            x_env = extract_variable_value_from_env(self.x, env)
            u_env = extract_variable_value_from_env(self.u, env)
            return linsys.evaluate_x_next(x_env, u_env)
        return sym.Evaluate(self.f, env)

class DTContinuousSystem:
    def __init__(self, f, x, u, initial_env=None, input_limits = None):
        '''
        Continuous dynamical system x_dot = f(x,u)
        :param f: A symbolic expression of the system dynamics.
        :param x: A list of symbolic variables. States.
        :param u: A list of symbolic variable. Inputs.
        :param initial_env: A dictionary "environment" specifying the initial state of the system
        :param input_limits: Input limits of the system
        '''
        self.dynamics = ContinuousDynamics(f,x,u)
        if input_limits is None:
            self.input_limits = np.vstack([np.full(u.shape[0], -1e9),np.full(u.shape[0], 1e9)])
        else:
            self.input_limits = input_limits
        self.u_bar = np.average(self.input_limits, axis=0)
        self.u_diff = np.diff(self.input_limits, axis=0)/2.
        if initial_env is None:
            self.env = {}
            for x_i in self.dynamics.x:
                self.env[x_i] = 0.
            for i,u_i in enumerate(self.dynamics.u):
                self.env[u_i]=self.u_bar[i]
        else:
            self.env = initial_env


    def forward_step(self, u=None, linearlize=False, modify_system=True, step_size = 1e-3, return_as_env = False, starting_state=None):
        if starting_state is not None:
            new_env = self._state_to_env(starting_state, u)  #just fill in the particular state and control input
        elif not modify_system:
            new_env = self.env.copy()
        else:
            new_env = self.env
        if u is not None:
            for i in range(u.shape[0]):
                new_env[self.dynamics.u[i]] = min(max(u[i],self.input_limits[0,i]),self.input_limits[1,i])
        else:
            for i in range(self.dynamics.u.shape[0]):
                new_env[self.dynamics.u[i]] = self.u_bar[i]
        delta_x = self.dynamics.evaluate_xdot(new_env, linearlize)*step_size
        #assign new xs
        for i in range(delta_x.shape[0]):
            new_env[self.dynamics.x[i]] += delta_x[i]
        if return_as_env:
            return new_env
        else:
            return extract_variable_value_from_env(self.dynamics.x, new_env)

    def get_reachable_polytopes(self, state, step_size = 1e-2, use_convex_hull=False):
        current_linsys = self.get_linearization(state, self.u_bar)
        x = np.ndarray.flatten(np.dot(current_linsys.A*step_size+np.eye(current_linsys.A.shape[0]),state))+\
            np.dot(current_linsys.B*step_size, self.u_bar)+np.ndarray.flatten(current_linsys.c*step_size)
        x = np.atleast_2d(x).reshape(-1,1)
        assert(len(x)==len(state))
        G = np.atleast_2d(np.dot(current_linsys.B*step_size, np.diag(self.u_diff)))
        if use_convex_hull:
            return convex_hull_of_point_and_polytope(state.reshape(x.shape),zonotope(x,G))
        return to_AH_polytope(zonotope(x,G))


    def get_linearization(self, state=None, u_bar = None, mode=None):
        if state is None:
            return self.dynamics.construct_linearized_system_at(self.env)
        else:
            env = self._state_to_env(state, u_bar)
            return self.dynamics.construct_linearized_system_at(env)

    def _state_to_env(self, state, u=None):
        env = {}
        # print('state',state)
        for i, s_i in enumerate(state):
            env[self.dynamics.x[i]] = s_i  #i^th component of state
        if u is None:
            for i, u_i in enumerate(self.dynamics.u):
                env[u_i] = self.u_bar[i]
        else:
            for i, u_i in enumerate(u):
                env[self.dynamics.u[i]] = u[i] # i^th control input
        return env

    def get_current_state(self):
        return extract_variable_value_from_env(self.dynamics.x, self.env)

def in_mode(c_i, env):
    for c_ij in c_i:
        if c_ij.Evaluate(env) is False:
            return False
    return True

class DTHybridSystem:
    def __init__(self, f_list, f_type_list, x, u, c_list, initial_env=None, input_limits=None):
        '''
        Hybrid system with multiple dynamics modes
        :param f_list: numpy array of system dynamics modes
        :param x: pydrake symbolic variables representing the states
        :param u: pydrake symbolic variables representing the inputs
        :param c_list: numpy array of Pydrake symbolic expressions c_i(x,u) describing when the system belong in that mode.
                        c_i(x,u) is a vector of functions describing the mode. All of c_i(x,u) >= 0 when system is in mode i.
                        Modes should be complete and mutually exclusive.
        :param initial_env: A dictionary "environment" specifying the initial state of the system
        :param input_limits: Input limits of the system
        '''
        assert f_list.shape[0] == c_list.shape[0]
        self.mode_count = f_list.shape[0]
        self.x = x
        self.u = u
        dynamics_list = []
        for i, f in enumerate(f_list):
            if f_type_list[i] == 'continuous':
                dynamics_list.append(ContinuousDynamics(f,self.x,self.u))
            elif f_type_list[i] == 'discrete':
                dynamics_list.append(DiscreteDynamics(f, self.x, self.u))
            else:
                raise ValueError
        self.dynamics_list = np.asarray(dynamics_list)

        if input_limits is None:
            self.input_limits = np.vstack([np.full(u.shape[0], -1e9),np.full(u.shape[0], 1e9)])
        else:
            self.input_limits = input_limits
        self.u_bar = np.atleast_2d((self.input_limits[1,:]+self.input_limits[0,:])/2.)
        self.u_diff = np.atleast_2d((self.input_limits[1,:]-self.input_limits[0,:])/2.)
        if initial_env is None:
            self.env = {}
            for x_i in self.x:
                self.env[x_i] = 0.
            for i,u_i in enumerate(self.u):
                self.env[u_i]=self.u_bar[i]
        else:
            self.env = initial_env
        self.c_list = c_list
        # Check the mode the system is in
        self.current_mode = -1
        #TODO

    def do_internal_updates(self):
        pass

    def forward_step(self, u=None, linearlize=False, modify_system=True, step_size = 1e-3, return_as_env = False,
                     return_mode = False, starting_state=None):
        if starting_state is not None:
            new_env = self._state_to_env(starting_state, u)
        elif not modify_system:
            new_env = self.env.copy()
        else:
            new_env = self.env
        if u is not None:
            for i in range(u.shape[0]):
                new_env[self.u[i]] = min(max(u[i],self.input_limits[0,i]),self.input_limits[1,i])
        else:
            for i in range(self.u.shape[0]):
                #ensure u is scalar
                new_env[self.u[i]] = np.ndarray.flatten(np.atleast_1d(self.u_bar))[i]
        # Check for which mode the system is in
        delta_x = None
        x_new = None
        mode = -1
        for i, c_i in enumerate(self.c_list):
            is_in_mode = in_mode(c_i,new_env)
            if not is_in_mode:
                continue
            if self.dynamics_list[i].type == 'continuous':
                delta_x = self.dynamics_list[i].evaluate_xdot(new_env, linearlize)*step_size
            elif self.dynamics_list[i].type == 'discrete':
                x_new = self.dynamics_list[i].evaluate_x_next(new_env, linearlize)
            else:
                raise ValueError
            mode = i
            break
        assert(mode != -1) # The system should always be in one mode
        # print('mode', mode)
        # print(self.env)
        #FIXME: check if system is in 2 modes (illegal)

        #assign new xs
        if self.dynamics_list[mode].type=='continuous':
            for i in range(delta_x.shape[0]):
                new_env[self.x[i]] += delta_x[i]
        elif self.dynamics_list[mode].type=='discrete':
            for i in range(x_new.shape[0]):
                new_env[self.x[i]] = x_new[i]
        else:
            raise ValueError

        self.do_internal_updates()

        #return options
        if return_as_env and not return_mode:
            return new_env
        elif return_as_env and return_mode:
            return new_env, mode
        elif not return_as_env and not return_mode:
            return extract_variable_value_from_env(self.x, new_env)
        else:
            return extract_variable_value_from_env(self.x, new_env), mode

    def get_reachable_polytopes(self, state, step_size=1e-2, use_convex_hull=False):
        polytopes_list = []
        for mode, c_i in enumerate(self.c_list):
            # FIXME: better way to check if the mode is possible
            # Very naive check: if all-min and all-max input lead to not being in mode, assume state is not in mode
            lower_bound_env = self.env.copy()
            upper_bound_env = self.env.copy()
            unactuated_env = self.env.copy()
            for i, u_i in enumerate(self.u):
                lower_bound_env[u_i] = self.input_limits[0,i]
                upper_bound_env[u_i] = self.input_limits[1, i]
                unactuated_env[u_i] = 0
            for i, x_i in enumerate(self.x):
                lower_bound_env[x_i] = state[i]
                upper_bound_env[x_i] = state[i]
                unactuated_env[x_i] = state[i]

            if (not in_mode(c_i, lower_bound_env)) and (not in_mode(c_i, upper_bound_env)) and not in_mode(c_i, unactuated_env):
                # print('dropping mode %i' %mode)
                continue

            current_linsys = self.get_linearization(state, mode=mode)
            if current_linsys is None:
                # this should not happen?
                raise Exception
            u_bar = (self.input_limits[1, :] + self.input_limits[0, :]) / 2.
            u_diff = (self.input_limits[1, :] - self.input_limits[0, :]) / 2.
            # print(mode)
            #     print('A', current_linsys.A)
            # print('B', current_linsys.B)
            #     print('c', current_linsys.c)

            if self.dynamics_list[mode].type == 'continuous':
                x = np.ndarray.flatten(
                    np.dot(current_linsys.A * step_size + np.eye(current_linsys.A.shape[0]), state)) + \
                    np.dot(current_linsys.B * step_size, u_bar) + np.ndarray.flatten(current_linsys.c * step_size)
                x = np.atleast_2d(x).reshape(-1, 1)
                assert (len(x) == len(state))
                G = np.atleast_2d(np.dot(current_linsys.B * step_size, np.diag(u_diff)))

            elif self.dynamics_list[mode].type == 'discrete':
                x = np.ndarray.flatten(
                    np.dot(current_linsys.A, state)) + \
                    np.dot(current_linsys.B, u_bar) + np.ndarray.flatten(current_linsys.c)
                x = np.atleast_2d(x).reshape(-1, 1)
                assert (len(x) == len(state))
                G = np.atleast_2d(np.dot(current_linsys.B, np.diag(u_diff)))
                # print('x', x)
                # print('G', G)
            else:
                raise ValueError
            # if mode==1:
            #     print(G, x)
            if use_convex_hull:
                polytopes_list.append(convex_hull_of_point_and_polytope(state.reshape(x.shape), zonotope(x,G)))
            else:
                polytopes_list.append(to_AH_polytope(zonotope(x, G)))
        return np.asarray(polytopes_list)

    def get_linearization(self, state=None, u_bar = None, mode=None):
        if state is None:
            return self.dynamics_list[self.current_mode].construct_linearized_system_at(self.env)
        else:
            env = self._state_to_env(state, u_bar)
            if mode is not None:
                # FIXME: construct but don't ask questions?
                # assert in_mode(self.c_list[mode], env)
                return self.dynamics_list[mode].construct_linearized_system_at(env)
            for mode, c_i in enumerate(self.c_list):
                if in_mode(c_i, env):
                    return self.dynamics_list[mode].construct_linearized_system_at(env)
            print('Warning: state is not in any mode')
            return None

    def _state_to_env(self, state, u=None):
        env = {}
        # print('state',state)
        for i, s_i in enumerate(state):
            env[self.x[i]] = s_i
        if u is None:
            for i, u_i in enumerate(self.u):
                env[u_i] = self.u_bar[i]
        else:
            for i, u_i in enumerate(u):
                env[self.u[i]] = u[i]
        return env

    def get_current_state(self):
        return extract_variable_value_from_env(self.x, self.env)
