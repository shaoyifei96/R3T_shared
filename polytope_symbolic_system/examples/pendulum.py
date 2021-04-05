import pydrake.symbolic as sym
from polytope_symbolic_system.common.symbolic_system import *

class Pendulum(DTContinuousSystem):
    def __init__(self, m=1, m_l=0, l=1, g=1, b=0, initial_state = np.array([0,0]), input_limits =None):
        '''
        Single link pendulum with mass at the end of a rod
        :param m: mass at the end of the pendulum. I = m*l**2
        :param m_l: mass of the rod. I_l = m_l*l**2/12
        :param l: length of the rod.
        :param g: gravity constant.
        :param b: damping. tau_damp = -b*theta_dot
        '''
        self.m = m
        self.m_l = m_l
        self.l = l
        self.g = g
        self.b = b
        #symbolic variables
        self.x = np.array([sym.Variable('x_' + str(i)) for i in range(2)])
        self.u = np.array([sym.Variable('u_' + str(i)) for i in range(1)])
        self.env = {self.x[0]:initial_state[0], self.x[1]:initial_state[1]}
        self.I = self.m*self.l**2+self.m_l*self.l**2/12
        self.t = -(self.m*self.g*self.l*sym.sin(self.x[0])+self.m_l*self.g*self.l/2*sym.sin(self.x[0]))
        self.f = np.asarray([self.x[1],1/self.I*(self.t+self.u[0]-self.b*self.x[1])])

        #symbolic system
        DTContinuousSystem.__init__(self, self.f, self.x, self.u, self.env, input_limits)

