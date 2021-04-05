import pydrake.symbolic as sym
import numpy as np
from common.symbolic_system import *

class Hopper_2d(DTHybridSystem):
    def __init__(self, m=5, J=500, m_l=1, J_l=0.5, l1=0.0, l2=0.0, k_g=2e3, b_g=20, \
                 g=9.8, flight_step_size = 1e-2, contact_step_size = 5e-3, descend_step_size_switch_threshold=2e-2, \
                 ground_height_function=lambda x: 0, initial_state=np.asarray([0.,0.,0.,1.5,1.0,0.,0.,0.,0.,0.])):


        '''
        2D hopper with actuated piston at the end of the leg.
        The model of the hopper follows the one described in "Hopping in Legged Systems" (Raibert, 1984)
        '''
        self.m = m
        self.J = J
        self.m_l = m_l
        self.J_l = J_l
        self.l1 = l1
        self.l2 = l2
        self.k_g_y = k_g
        self.k_g_x = 2e3
        self.b_g_x = 200
        self.b_g = b_g
        self.g = g
        self.ground_height_function = ground_height_function
        self.r0 = 1.5
        # state machine for touchdown detection
        self.xTD = sym.Variable('xTD')
        self.was_in_contact = False

        # Symbolic variables
        # State variables are s = [x_ft, y_ft, theta, phi, r]
        # self.x = [s, sdot]
        # Inputs are self.u = [tau, chi]
        self.x = np.array([sym.Variable('x_' + str(i)) for i in range(10)])
        self.u = np.array([sym.Variable('u_' + str(i)) for i in range(2)])

        # Initial state
        self.initial_env = {}
        for i, state in enumerate(initial_state):
            self.initial_env[self.x[i]]=state
        self.initial_env[self.xTD] = 0
        self.k0 = 800
        self.b_leg = 2
        self.k0_stabilize = 40
        self.b0_stabilize = 10
        self.k0_restore = 60
        self.b0_restore = 15
        # self.flight_step_size = flight_step_size
        # self.contact_step_size = contact_step_size
        # self.descend_step_size_switch_threshold = descend_step_size_switch_threshold
        # self.hover_step_size_switch_threshold=-0.75
        # print(self.initial_env)

        # Dynamic modes
        Fx_contact = -self.k_g_x*(self.x[0]-self.xTD)-self.b_g_x*self.x[5]
        Fx_flight = 0.
        Fy_contact = -self.k_g_y*(self.x[1]-self.ground_height_function(self.x[0]))-self.b_g*self.x[6]*(1-np.exp(self.x[1]*16))
        Fy_flight = 0.

        R = self.x[4]-self.l1
        # EOM is obtained from Russ Tedrake's Thesis
        a1 = -self.m_l*R
        a2 = (self.J_l-self.m_l*R*self.l1)*sym.cos(self.x[2])
        b1 = self.m_l*R
        b2 = (self.J_l -self.m_l*R*self.l1)*sym.sin(self.x[2])
        c1 = self.m*R
        c2 = (self.J_l+self.m*R*self.x[4])*sym.cos(self.x[2])
        c3 = self.m*R*self.l2*sym.cos(self.x[3])
        c4 = self.m*R*sym.sin(self.x[2])
        d1 = -self.m*R
        d2 = (self.J_l+self.m*R*self.x[4])*sym.sin(self.x[2])
        d3 = self.m*R*self.l2*sym.sin(self.x[3])
        d4 = -self.m*R*sym.cos(self.x[2])
        e1 = self.J_l*self.l2*sym.cos(self.x[2]-self.x[3])
        e2 = -self.J*R
        self.b_r_ascend = 0.
        r_diff = self.x[4]-self.r0
        F_leg_flight = -self.k0_restore*r_diff-self.b0_restore*self.x[9]
        F_leg_ascend = -self.u[1]*r_diff - self.b_r_ascend * self.x[9]#self.u[1] * (1-np.exp(10*r_diff_upper)/(np.exp(10*r_diff_upper)+1))#+(- self.k0_stabilize * r_diff_upper - self.b0_stabilize * self.x[9])*(np.exp(10*r_diff_upper)/(np.exp(10*r_diff_upper)+1))
        F_leg_descend = -self.k0*r_diff-self.b_leg*self.x[9]
        # F_leg_descend = F_leg_ascend

        self.tau_p = 400.
        self.tau_d = 10.
        hip_x_dot = self.x[5]+self.x[9]*sym.sin(self.x[2])+self.x[4]*sym.cos(self.x[2])*self.x[7]
        hip_y_dot = self.x[6]+self.x[9]*sym.cos(self.x[2])-self.x[4]*sym.sin(self.x[2])*self.x[7]
        alpha_des_ascend = 0.6*sym.atan(hip_x_dot/(-hip_y_dot-1e-6))#-sym.atan(self.x[5]/self.x[6]) # point toward
        alpha_des_descend = 0.6*sym.atan(hip_x_dot/(hip_y_dot+1e-6)) # point toward landing point
        tau_leg_flight_ascend = (self.tau_p*(alpha_des_ascend-self.x[2])-self.tau_d*self.x[7])*-1
        tau_leg_flight_descend = (self.tau_p*(alpha_des_descend-self.x[2])-self.tau_d*self.x[7])*-1
        tau_leg_contact = self.u[0]

        def get_ddots(Fx, Fy, F_leg, u0):
            alpha = (self.l1*Fy*sym.sin(self.x[2])-self.l1*Fx*sym.cos(self.x[2])-u0)
            A = sym.cos(self.x[2])*alpha-R*(Fx-F_leg*sym.sin(self.x[2])-self.m_l*self.l1*self.x[7]**2*sym.sin(self.x[2]))
            B = sym.sin(self.x[2])*alpha+R*(self.m_l*self.l1*self.x[7]**2*sym.cos(self.x[2])+Fy-F_leg*sym.cos(self.x[2])-self.m_l*self.g)
            C = sym.cos(self.x[2])*alpha+R*F_leg*sym.sin(self.x[2])+self.m*R*(self.x[4]*self.x[7]**2*sym.sin(self.x[2])+self.l2*self.x[8]**2*sym.sin(self.x[3])-2*self.x[9]*self.x[7]*sym.cos(self.x[2]))
            D = sym.sin(self.x[2])*alpha-R*(F_leg*sym.cos(self.x[2])-self.m*self.g)-self.m*R*(2*self.x[9]*self.x[7]*sym.sin(self.x[2])+self.x[4]*self.x[7]**2*sym.cos(self.x[2])+self.l2*self.x[8]**2*sym.cos(self.x[3]))
            E = self.l2*sym.cos(self.x[2]-self.x[3])*alpha-R*(self.l2*F_leg*sym.sin(self.x[3]-self.x[2])+u0)

            return np.asarray([(A*b1*c2*d4*e2 - A*b1*c3*d4*e1 - A*b1*c4*d2*e2 + A*b1*c4*d3*e1 + A*b2*c4*d1*e2 - B*a2*c4*d1*e2 - C*a2*b1*d4*e2 + D*a2*b1*c4*e2 + E*a2*b1*c3*d4 - E*a2*b1*c4*d3)/(a1*b1*c2*d4*e2 - a1*b1*c3*d4*e1 - a1*b1*c4*d2*e2 + a1*b1*c4*d3*e1 + a1*b2*c4*d1*e2 - a2*b1*c1*d4*e2), \
                               (A*b2*c1*d4*e2 + B*a1*c2*d4*e2 - B*a1*c3*d4*e1 - B*a1*c4*d2*e2 + B*a1*c4*d3*e1 - B*a2*c1*d4*e2 - C*a1*b2*d4*e2 + D*a1*b2*c4*e2 + E*a1*b2*c3*d4 - E*a1*b2*c4*d3)/(a1*b1*c2*d4*e2 - a1*b1*c3*d4*e1 - a1*b1*c4*d2*e2 + a1*b1*c4*d3*e1 + a1*b2*c4*d1*e2 - a2*b1*c1*d4*e2), \
                               -(A*b1*c1*d4*e2 - B*a1*c4*d1*e2 - C*a1*b1*d4*e2 + D*a1*b1*c4*e2 + E*a1*b1*c3*d4 - E*a1*b1*c4*d3)/(a1*b1*c2*d4*e2 - a1*b1*c3*d4*e1 - a1*b1*c4*d2*e2 + a1*b1*c4*d3*e1 + a1*b2*c4*d1*e2 - a2*b1*c1*d4*e2), \
                               (A*b1*c1*d4*e1 - B*a1*c4*d1*e1 - C*a1*b1*d4*e1 + D*a1*b1*c4*e1 + E*a1*b1*c2*d4 - E*a1*b1*c4*d2 + E*a1*b2*c4*d1 - E*a2*b1*c1*d4)/(a1*b1*c2*d4*e2 - a1*b1*c3*d4*e1 - a1*b1*c4*d2*e2 + a1*b1*c4*d3*e1 + a1*b2*c4*d1*e2 - a2*b1*c1*d4*e2), \
                               (A*b1*c1*d2*e2 - A*b1*c1*d3*e1 - A*b2*c1*d1*e2 - B*a1*c2*d1*e2 + B*a1*c3*d1*e1 + B*a2*c1*d1*e2 - C*a1*b1*d2*e2 + C*a1*b1*d3*e1 + C*a1*b2*d1*e2 + D*a1*b1*c2*e2 - D*a1*b1*c3*e1 - D*a2*b1*c1*e2 - E*a1*b1*c2*d3 + E*a1*b1*c3*d2 - E*a1*b2*c3*d1 + E*a2*b1*c1*d3)/(a1*b1*c2*d4*e2 - a1*b1*c3*d4*e1 - a1*b1*c4*d2*e2 + a1*b1*c4*d3*e1 + a1*b2*c4*d1*e2 - a2*b1*c1*d4*e2)])

        flight_ascend_dynamics = np.hstack((self.x[5:], get_ddots(Fx_flight, Fy_flight, F_leg_flight, tau_leg_flight_ascend)))
        flight_descend_dynamics = np.hstack((self.x[5:], get_ddots(Fx_flight, Fy_flight, F_leg_flight, tau_leg_flight_descend)))
        contact_descend_dynamics = np.hstack((self.x[5:], get_ddots(Fx_contact, Fy_contact, F_leg_descend, tau_leg_contact)))
        contact_ascend_dynamics = np.hstack((self.x[5:], get_ddots(Fx_contact, Fy_contact, F_leg_ascend, tau_leg_contact)))

        flight_ascend_conditions = np.asarray([self.x[1] > self.ground_height_function(self.x[0]), hip_y_dot>0])
        flight_descend_conditions = np.asarray([self.x[1] > self.ground_height_function(self.x[0]), hip_y_dot<=0])
        contact_descend_coditions = np.asarray([self.x[1] <= self.ground_height_function(self.x[0]), self.x[9] < 0])
        contact_ascend_coditions = np.asarray([self.x[1] <= self.ground_height_function(self.x[0]), self.x[9] >= 0])

        self.f_list = np.asarray([flight_ascend_dynamics, flight_descend_dynamics, contact_descend_dynamics, contact_ascend_dynamics])
        self.f_type_list = np.asarray(['continuous', 'continuous', 'continuous','continuous'])
        self.c_list = np.asarray([flight_ascend_conditions, flight_descend_conditions, contact_descend_coditions, contact_ascend_coditions])

        DTHybridSystem.__init__(self, self.f_list, self.f_type_list, self.x, self.u, self.c_list, \
                                self.initial_env, input_limits=np.vstack([[-500,1.4e3], [500,2.5e3]]))

    def get_cg_coordinate_states(self, env = None):
        """
        Convert the state into the representation used in MIT 6.832 PSet4
        [x2, y2, theta2, theta1-theta2, w]
        :param env:
        :return:
        """
        if env is None:
            env = self.env
        # extract variables from the environment
        theta1 = env[self.x[0]]
        theta2 = env[self.x[1]]
        x0 = env[self.x[2]]
        y0 = env[self.x[3]]
        w = env[self.x[4]]
        theta1_dot = env[self.x[5]]
        theta2_dot = env[self.x[6]]
        x0_dot = env[self.x[7]]
        y0_dot = env[self.x[8]]
        w_dot =  env[self.x[9]]

        # compute forward kinematics
        x1 = x0+self.r1*np.sin(theta1)
        y1 = y0+self.r1*np.cos(theta1)
        x2 = x0+w*np.sin(theta1)+self.r2*np.cos(theta2)
        y2 = y0+w*np.cos(theta1)+self.r2*np.sin(theta2)

        # compute derivatives
        x1_dot = x0_dot+self.r1*np.cos(theta1)*theta1_dot
        y1_dot = y0_dot-self.r1*np.sin(theta1)*theta1_dot
        x2_dot = x0_dot+w_dot*np.sin(theta1)+w*np.cos(theta1)*theta1_dot+self.r2*np.cos(theta2)*theta2_dot
        y2_dot = y1_dot+w_dot*np.cos(theta1)-w*np.sin(theta1)*theta1_dot-self.r2*np.sin(theta2)*theta2_dot

        return x2, y2, theta2, theta1-theta2, w, x2_dot, y2_dot, theta2_dot, theta1_dot-theta2_dot, w_dot

    def do_internal_updates(self):
        # extract variables from the environment
        try:
            x0 = self.env[self.x[0]][0]
        except:
            x0 = self.env[self.x[0]]
        try:
            y0 = self.env[self.x[1]][0]
        except:
            y0 = self.env[self.x[1]]
        if not self.was_in_contact and y0-self.ground_height_function(x0)<=0:
            # just touched down
            # set the touchdown point
            self.env[self.xTD] = x0
        self.was_in_contact=(y0-self.ground_height_function(x0)<=0)
        #FIXME: _state_to_env does not set self.env[self.xTD]

    def _state_to_env(self, state, u=None):
        env = {}
        for i, s_i in enumerate(state):
            env[self.x[i]] = s_i
        if u is None:
            for u_i in self.u:
                env[u_i] = 0
        else:
            for i, u_i in enumerate(u):
                env[self.u[i]] = u[i]
        # FIXME: This is sketchy
        env[self.xTD] = state[0]
        return env

    # def get_reachable_polytopes(self, state, step_size=1e-2, use_convex_hull=True):
    #     polytopes_list = []
    #     for mode, c_i in enumerate(self.c_list):
    #         # FIXME: better way to check if the mode is possible
    #         # Very naive check: if all-min and all-max input lead to not being in mode, assume state is not in mode
    #         lower_bound_env = self.env.copy()
    #         upper_bound_env = self.env.copy()
    #         unactuated_env = self.env.copy()
    #         for i, u_i in enumerate(self.u):
    #             lower_bound_env[u_i] = self.input_limits[0,i]
    #             upper_bound_env[u_i] = self.input_limits[1, i]
    #             unactuated_env[u_i] = 0
    #         for i, x_i in enumerate(self.x):
    #             lower_bound_env[x_i] = state[i]
    #             upper_bound_env[x_i] = state[i]
    #             unactuated_env[x_i] = state[i]
    #
    #         if (not in_mode(c_i, lower_bound_env)) and (not in_mode(c_i, upper_bound_env)) and not in_mode(c_i, unactuated_env):
    #             # print('dropping mode %i' %mode)
    #             continue
    #
    #         current_linsys = self.get_linearization(state, mode=mode)
    #         if current_linsys is None:
    #             # this should not happen?
    #             raise Exception
    #         u_bar = (self.input_limits[1, :] + self.input_limits[0, :]) / 2
    #         u_diff = (self.input_limits[1, :] - self.input_limits[0, :]) / 2
    #         print('ubar, udiff', u_bar, u_diff)
    #         # print(mode)
    #         #     print('A', current_linsys.A)
    #         # print('B', current_linsys.B)
    #         #     print('c', current_linsys.c)
    #         # t = ((state[6]**2+2*self.g*abs(state[1]-self.ground_height_function(state[0])))**0.5-abs(state[6]))/self.g
    #         # # print((0>=state[1]-self.ground_height_function(state[0])>=self.hover_step_size_switch_threshold),state[6])
    #         #
    #         # if (0>=state[1]-self.ground_height_function(state[0])>=self.hover_step_size_switch_threshold):# and abs(state[6])<0.7:
    #         #     variable_step_size=min(max(self.contact_step_size, (0.65-state[4])/state[9]), self.flight_step_size)
    #         #     # print(variable_step_size)
    #         # elif (0<state[1] - self.ground_height_function(state[0]) < self.descend_step_size_switch_threshold and state[6] < 0) or\
    #         #         (state[1]-self.ground_height_function(state[0])<=self.hover_step_size_switch_threshold):
    #         #     #descending to ground
    #         #     # if emerging from the ground, decrease step size further
    #         #     # print((0.05-(state[1]-self.ground_height_function(state[0])))/state[6]))
    #         #     # variable_step_size = max(min(self.contact_step_size, (0.05-(state[1]-self.ground_height_function(state[0])))/state[6]), 1e-3)f
    #         #     variable_step_size = self.contact_step_size
    #         #     # print('using contact step size')
    #         # elif self.flight_step_size>1.2*t and state[6]<0 and (state[1]-self.ground_height_function(state[0])>0.):
    #         #     # descending in flight
    #         #     variable_step_size=max(self.contact_step_size, t)
    #         #     # print('using adaptive flight step size', variable_step_size)
    #         # else:
    #         #     variable_step_size = self.flight_step_size
    #
    #         # override
    #         variable_step_size = 1e-3
    #         if self.dynamics_list[mode].type == 'continuous':
    #             x = np.ndarray.flatten(
    #                 np.dot(current_linsys.A * variable_step_size + np.eye(current_linsys.A.shape[0]), state)) + \
    #                 np.dot(current_linsys.B * variable_step_size, u_bar) + np.ndarray.flatten(current_linsys.c * variable_step_size)
    #             x = np.atleast_2d(x).reshape(-1, 1)
    #             assert (len(x) == len(state))
    #             G = np.atleast_2d(np.dot(current_linsys.B * variable_step_size, np.diag(u_diff)))
    #         #
    #         # elif self.dynamics_list[mode].type == 'discrete':
    #         #     x = np.ndarray.flatten(
    #         #         np.dot(current_linsys.A, state)) + \
    #         #         np.dot(current_linsys.B, u_bar) + np.ndarray.flatten(current_linsys.c)
    #         #     x = np.atleast_2d(x).reshape(-1, 1)
    #         #     assert (len(x) == len(state))
    #         #     G = np.atleast_2d(np.dot(current_linsys.B, np.diag(u_diff)))
    #         #     # print('x', x)
    #         #     # print('G', G)
    #         else:
    #             raise ValueError
    #         # if mode==1:
    #         #     print(G, x)
    #         if use_convex_hull:
    #             polytopes_list.append(convex_hull_of_point_and_polytope(x, zonotope(x,G)))
    #         else:
    #             polytopes_list.append(to_AH_polytope(zonotope(x, G)))
    #     return np.asarray(polytopes_list)
    #
    # def forward_step(self, u=None, linearlize=False, modify_system=True, step_size = 1e-3, return_as_env = False,
    #                  return_mode = False, starting_state=None):
    #     if starting_state is not None:
    #         new_env = self._state_to_env(starting_state, u)
    #     elif not modify_system:
    #         new_env = self.env.copy()
    #     else:
    #         new_env = self.env
    #     if u is not None:
    #         for i in range(u.shape[0]):
    #             new_env[self.u[i]] = min(max(u[i],self.input_limits[0,i]),self.input_limits[1,i])
    #     else:
    #         for i in range(self.u.shape[0]):
    #             new_env[self.u[i]] = 0
    #     # Check for which mode the system is in
    #     delta_x = None
    #     x_new = None
    #     mode = -1
    #     for i, c_i in enumerate(self.c_list):
    #         is_in_mode = in_mode(c_i,new_env)
    #         if not is_in_mode:
    #             continue
    #         if self.dynamics_list[i].type == 'continuous':
    #             # use small step for contact
    #             # if new_env[self.x[1]]<2e-1:
    #             #     variable_step_size = 1e-3
    #             # else:
    #             #     variable_step_size = 1e-1
    #             delta_x = self.dynamics_list[i].evaluate_xdot(new_env, linearlize)*step_size
    #         # elif self.dynamics_list[i].type == 'discrete':
    #         #     x_new = self.dynamics_list[i].evaluate_x_next(new_env, linearlize)
    #         else:
    #             raise ValueError
    #         mode = i
    #         break
    #     # assert(mode != -1) # The system should always be in one mode
    #     # print('mode', mode)
    #     # print(self.env)
    #     #FIXME: check if system is in 2 modes (illegal)
    #
    #     #assign new xs
    #     if self.dynamics_list[mode].type=='continuous':
    #         for i in range(delta_x.shape[0]):
    #             new_env[self.x[i]] += delta_x[i]
    #     # elif self.dynamics_list[mode].type=='discrete':
    #     #     for i in range(x_new.shape[0]):
    #     #         new_env[self.x[i]] = x_new[i]
    #     else:
    #         raise ValueError
    #
    #     # if new_env[self.x[4]] < self.r_min or new_env[self.x[4]]>self.r_max:
    #     #     new_env[self.x[4]] = min(max(new_env[self.x[4]], self.r_min), self.r_max)
    #     #     new_env[self.x[9]] = 0
    #     self.do_internal_updates()
    #
    #     #return options
    #     if return_as_env and not return_mode:
    #         return new_env
    #     elif return_as_env and return_mode:
    #         return new_env, mode
    #     elif not return_as_env and not return_mode:
    #         return extract_variable_value_from_env(self.x, new_env)
    #     else:
    #         return extract_variable_value_from_env(self.x, new_env), mode
    #
