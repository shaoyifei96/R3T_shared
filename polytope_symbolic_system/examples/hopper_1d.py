import pydrake.symbolic as sym
from common.symbolic_system import *

class Hopper_1d(DTHybridSystem):
    def __init__(self, m=1, l=1, p=0.1, b=0.85, g=9.8, f_max=80, initial_state = None, epsilon = 1e-7):
        '''
        Vertical 1D hopper with actuated piston at the end of the leg.
        The hopper has 3 dynamics mode decided by the body height h:
        1. h>l+p: free flight. xddot = -g
        2. l<h<=l+p: piston in contact with the ground and the hopper may push itself up.
            xddot = f/m-g.
        3. h<=l: piston is fully retracted. The hopper bounces off the ground
            in an elastic collision with xdot(n+1) = -b*xdot(n), where 0<b<1

        :param m: mass of the hopper
        :param l: leg length of the hopper
        :param p: piston length of the hopper
        :param b: damping factor fo the ground
        :param g: gravity constant
        :param f_range: maximum force the piston can exert
        :param epsilon: for maintaining stability near mode switches
        '''
        self.m = m
        self.l = l
        self.p = p
        self.b = b
        self.g = g
        self.f_max = f_max
        self.epsilon = epsilon
        if initial_state is None:
            initial_state = np.asarray([self.l+self.p, 0])

        # Symbolic variables
        # x = [x, xdot], upward positive, floor is x<=0
        self.x = np.array([sym.Variable('x_' + str(i)) for i in range(2)])
        self.u = np.array([sym.Variable('u_' + str(i)) for i in range(1)])

        self.initial_env = {self.x[0]: initial_state[0], self.x[1]:initial_state[1]}

        # parameter sanity checks
        assert(self.m>0)
        assert(self.l>0)
        assert(self.p>0)
        assert(self.g>0)
        assert(0<self.b<1)
        assert(self.f_max >= 0)
        assert(self.l<self.initial_env[self.x[0]])


        # Dynamic modes
        # free flight
        free_flight_dynamics = np.asarray([self.x[1], -self.g])
        # piston contact
        piston_contact_dynamics = np.asarray([self.x[1], self.u[0]/self.m-self.g])
        # piston retract
        piston_retracted_dynamics = np.asarray([self.l+self.epsilon, -self.x[1]*self.b])
        self.f_list = np.asarray([free_flight_dynamics, piston_contact_dynamics, piston_retracted_dynamics])
        self.f_type_list = np.asarray(['continuous', 'continuous', 'discrete'])

        # contact mode conditions
        free_flight_conditions = np.asarray([self.x[0]>self.l+self.p])
        piston_contact_conditions = np.asarray([self.l<self.x[0], self.x[0]<=self.l+self.p])
        piston_retracted_conditions = np.asarray([self.x[0]<=self.l])
        self.c_list = np.asarray([free_flight_conditions, piston_contact_conditions, piston_retracted_conditions])

        DTHybridSystem.__init__(self, self.f_list, self.f_type_list, self.x, self.u, self.c_list, \
                                self.initial_env, input_limits=np.asarray([[0], [self.f_max]]))
