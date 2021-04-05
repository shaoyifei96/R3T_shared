import pydrake.symbolic as sym
import numpy as np
from common.symbolic_system import DTHybridSystem

class Hopper_2d(DTHybridSystem):
    def __init__(self, M1=1., I1=1., M2=10., I2=10., r1=0.5, r2=0.4, KL=1e3, KL2=1e5, BL2=125, KG=1e4, BG=75, k0=1., \
                 g=9.8, ground_height_function=lambda x: 0, initial_state=np.asarray([0.,0.,0.,1.5,1.0,0.,0.,0.,0.,0.])):


        '''
        2D hopper with actuated piston at the end of the leg.
        The model of the hopper follows the one described in "Hopping in Legged Systems" (Raibert, 1984)
        '''
        self.M1 = M1
        self.I1 = I1
        self.M2 = M2
        self.I2 = I2
        self.r1 = r1
        self.r2 = r2
        self.KL = KL
        self.KL2 = KL2
        self.BL2 = BL2
        self.k0 = k0
        self.KG = KG
        self.BG = BG
        self.g = g
        self.ground_height_function = ground_height_function

        # state machine for touchdown detection
        self.xTD = sym.Variable('xTD')
        self.was_in_contact = False

        # Symbolic variables
        # State variables are s = [theta1, theta2, x0, y0, w]
        # self.x = [s, sdot]
        # Inputs are self.u = [tau, chi]
        self.x = np.array([sym.Variable('x_' + str(i)) for i in range(10)])
        self.u = np.array([sym.Variable('u_' + str(i)) for i in range(2)])

        # Initial state
        self.initial_env = {}
        for i, state in enumerate(initial_state):
            self.initial_env[self.x[i]]=state
        self.initial_env[self.xTD] = 0
        # print(self.initial_env)

        # Dynamic modes
        FK_extended = self.KL*(self.k0-self.x[4]+self.u[1])
        FK_retracted = self.KL2*(self.k0-self.x[4]+self.u[1])-self.BL2*self.x[9]
        FX_contact = -self.KG*(self.x[2]-self.xTD)-self.BG*self.x[7] #FIXME: handling xTD
        FX_flight = 0.
        FY_contact = -self.KG*(self.x[3]-self.ground_height_function(self.x[2]))-self.BG*(self.x[8])
        FY_flight = 0.

        W = self.x[4]-self.r1

        # EOM is obtained from Raibert 1984 paper, Appendix I eqn 40~44
        # The EOM is in the following form
        # a1*theta1_ddot + a2*theta2_ddot + a3*x0_ddot + a4*w_ddot = A
        # b1*theta1_ddot + b2*theta2_ddot + b3*y0_ddot + b4*w_ddot = B
        # c1*theta1_ddot + c2*x0_ddot = C
        # d1*theta1_ddot + d2*y0_ddot = D
        # e1*theta1_ddot + e2*theta2_ddot = E

        a1 = sym.cos(self.x[0]) * (self.M2 * W * self.x[4] + self.I1)
        a2 = self.M2 * self.r2 * W * sym.cos(self.x[1])
        a3 = self.M2*W
        a4 = self.M2 * W * sym.sin(self.x[0])

        b1 = -sym.sin(self.x[0]) * (self.M2 * W * self.x[4] + self.I1)
        b2 = -self.M2 * self.r2 * W * sym.sin(self.x[1])
        b3 = self.M2*W
        b4 = self.M2 * W * sym.cos(self.x[1])

        c1 = sym.cos(self.x[0]) * (self.M1 * self.r1 * W - self.I1)
        c2 = self.M1*W

        d1 = -sym.sin(self.x[0]) * (self.M1 * self.r1 * W - self.I1)
        d2 = self.M1*W

        e1 = -sym.cos(self.x[1] - self.x[0]) * self.I1 * self.r2
        e2 = self.I2*self.x[4]

        # free flight with extended leg
        flight_extended_conditions = np.asarray([self.k0-self.x[4]+self.u[1]>0,
                                                      self.x[3]-self.ground_height_function(self.x[2])>0])

        A_flight_extended = W * self.M2 * (self.x[5] ** 2 * W * sym.sin(self.x[0]) - 2 * self.x[5] * self.x[9] * sym.cos(self.x[0]) + \
                                           self.r2 * self.x[6] ** 2 * sym.sin(self.x[1]) + self.r1 * self.x[5] ** 2 * sym.sin(self.x[0])) - \
                            self.r1 * FX_flight * (sym.cos(self.x[0]) ** 2) + sym.cos(self.x[0]) * (self.r1 * FY_flight * sym.sin(self.x[0]) - self.u[0]) + \
                            FK_extended * W * sym.sin(self.x[0])

        B_flight_extended = W * self.M2 * (self.x[5] ** 2 * W * sym.cos(self.x[0]) + 2 * self.x[5] * self.x[9] * sym.sin(self.x[0]) + \
                                           self.r2 * self.x[6] ** 2 * sym.cos(self.x[1]) + self.r1 * self.x[5] ** 2 * sym.cos(self.x[0]) - g) + \
                            self.r1 * FX_flight * sym.cos(self.x[0]) * sym.sin(self.x[0]) - sym.sin(self.x[0]) * (self.r1 * FY_flight * sym.sin(self.x[0]) - self.u[0]) + \
                            FK_extended * W * sym.cos(self.x[0])

        C_flight_extended = W * (self.M1 * self.r1 * self.x[5] ** 2 * sym.sin(self.x[0]) - FK_extended * sym.sin(self.x[0]) + FX_flight) - \
                            sym.cos(self.x[0]) * (FY_flight * self.r1 * sym.sin(self.x[0]) - FX_flight * self.r1 * sym.cos(self.x[0]) - self.u[0])

        D_flight_extended = W * (self.M1 * self.r1 * self.x[5] ** 2 * sym.cos(self.x[0]) - FK_extended * sym.cos(self.x[0]) + FY_flight - self.M1 * self.g) - \
                            sym.sin(self.x[0]) * (FY_flight * self.r1 * sym.sin(self.x[0]) - FX_flight * self.r1 * sym.cos(self.x[0]) - self.u[0])

        E_flight_extended = W * (FK_extended * self.r2 * sym.sin(self.x[1] - self.x[0]) + self.u[0]) - self.r2 * sym.cos(self.x[1] - self.x[0]) * \
                            (self.r1 * FY_flight * sym.sin(self.x[0]) - self.r1 * FX_flight * sym.cos(self.x[0]) - self.u[0])

        theta1_ddot_flight_extended = ((a4*b2/b4-a2)*E_flight_extended/e2-a3*C_flight_extended/c2+a4*b3*D_flight_extended/(b4*d2)+\
                                      A_flight_extended-a4*B_flight_extended/b4)/\
                                      (a1-b1*a4/b4-(a2-a4*b2/b4)*e1/e2-a3*c1/c2+a4*b3*d1/(b4*d2))

        theta2_ddot_flight_extended = (E_flight_extended-e1*theta1_ddot_flight_extended)/e2

        x_ddot_flight_extended = (C_flight_extended-c1*theta1_ddot_flight_extended)/c2

        y_ddot_flight_extended = (D_flight_extended-d1*theta1_ddot_flight_extended)/d2

        w_ddot_flight_extended = (A_flight_extended+B_flight_extended-(a1+b1)*theta1_ddot_flight_extended-(a2+b2)*theta2_ddot_flight_extended-\
                                  a3*x_ddot_flight_extended-b3*y_ddot_flight_extended)/(a4+b4)

        flight_extended_ddots = np.asarray([theta1_ddot_flight_extended, theta2_ddot_flight_extended, x_ddot_flight_extended, y_ddot_flight_extended, w_ddot_flight_extended])

        flight_extended_dynamics = np.hstack((self.x[5:], flight_extended_ddots))

        # free flight with retracted leg
        flight_retracted_conditions = np.asarray([self.k0-self.x[4]+self.u[1]<=0,
                                                      self.x[3]-self.ground_height_function(self.x[2])>0])

        A_flight_retracted = W * self.M2 * (self.x[5] ** 2 * W * sym.sin(self.x[0]) - 2 * self.x[5] * self.x[9] * sym.cos(self.x[0]) + \
                                           self.r2 * self.x[6] ** 2 * sym.sin(self.x[1]) + self.r1 * self.x[5] ** 2 * sym.sin(self.x[0])) - \
                            self.r1 * FX_flight * (sym.cos(self.x[0]) ** 2) + sym.cos(self.x[0]) * (self.r1 * FY_flight * sym.sin(self.x[0]) - self.u[0]) + \
                            FK_retracted * W * sym.sin(self.x[0])

        B_flight_retracted = W * self.M2 * (self.x[5] ** 2 * W * sym.cos(self.x[0]) + 2 * self.x[5] * self.x[9] * sym.sin(self.x[0]) + \
                                           self.r2 * self.x[6] ** 2 * sym.cos(self.x[1]) + self.r1 * self.x[5] ** 2 * sym.cos(self.x[0]) - g) + \
                            self.r1 * FX_flight * sym.cos(self.x[0]) * sym.sin(self.x[0]) - sym.sin(self.x[0]) * (self.r1 * FY_flight * sym.sin(self.x[0]) - self.u[0]) + \
                            FK_retracted * W * sym.cos(self.x[0])

        C_flight_retracted = W * (self.M1 * self.r1 * self.x[5] ** 2 * sym.sin(self.x[0]) - FK_retracted * sym.sin(self.x[0]) + FX_flight) - \
                            sym.cos(self.x[0]) * (FY_flight * self.r1 * sym.sin(self.x[0]) - FX_flight * self.r1 * sym.cos(self.x[0]) - self.u[0])

        D_flight_retracted = W * (self.M1 * self.r1 * self.x[5] ** 2 * sym.cos(self.x[0]) - FK_retracted * sym.cos(self.x[0]) + FY_flight - self.M1 * self.g) - \
                            sym.sin(self.x[0]) * (FY_flight * self.r1 * sym.sin(self.x[0]) - FX_flight * self.r1 * sym.cos(self.x[0]) - self.u[0])

        E_flight_retracted = W * (FK_retracted * self.r2 * sym.sin(self.x[1] - self.x[0]) + self.u[0]) - self.r2 * sym.cos(self.x[1] - self.x[0]) * \
                            (self.r1 * FY_flight * sym.sin(self.x[0]) - self.r1 * FX_flight * sym.cos(self.x[0]) - self.u[0])

        theta1_ddot_flight_retracted = ((a4*b2/b4-a2)*E_flight_retracted/e2-a3*C_flight_retracted/c2+a4*b3*D_flight_retracted/(b4*d2)+\
                                      A_flight_retracted-a4*B_flight_retracted/b4)/\
                                      (a1-b1*a4/b4-(a2-a4*b2/b4)*e1/e2-a3*c1/c2+a4*b3*d1/(b4*d2))

        theta2_ddot_flight_retracted = (E_flight_retracted-e1*theta1_ddot_flight_retracted)/e2

        x_ddot_flight_retracted = (C_flight_retracted-c1*theta1_ddot_flight_retracted)/c2

        y_ddot_flight_retracted = (D_flight_retracted-d1*theta1_ddot_flight_retracted)/d2

        w_ddot_flight_retracted = (A_flight_retracted+B_flight_retracted-(a1+b1)*theta1_ddot_flight_retracted-(a2+b2)*theta2_ddot_flight_retracted-\
                                  a3*x_ddot_flight_retracted-b3*y_ddot_flight_retracted)/(a4+b4)

        flight_retracted_ddots = np.asarray([theta1_ddot_flight_retracted, theta2_ddot_flight_retracted, x_ddot_flight_retracted, y_ddot_flight_retracted, w_ddot_flight_retracted])

        flight_retracted_dynamics = np.hstack((self.x[5:], flight_retracted_ddots))

        # contact with extended leg
        contact_extended_conditions = np.asarray([self.k0-self.x[4]+self.u[1]>0,
                                                      self.x[3]-self.ground_height_function(self.x[2])<=0])

        A_contact_extended = W * self.M2 * (self.x[5] ** 2 * W * sym.sin(self.x[0]) - 2 * self.x[5] * self.x[9] * sym.cos(self.x[0]) + \
                                           self.r2 * self.x[6] ** 2 * sym.sin(self.x[1]) + self.r1 * self.x[5] ** 2 * sym.sin(self.x[0])) - \
                            self.r1 * FX_contact * (sym.cos(self.x[0]) ** 2) + sym.cos(self.x[0]) * (self.r1 * FY_contact * sym.sin(self.x[0]) - self.u[0]) + \
                            FK_extended * W * sym.sin(self.x[0])

        B_contact_extended = W * self.M2 * (self.x[5] ** 2 * W * sym.cos(self.x[0]) + 2 * self.x[5] * self.x[9] * sym.sin(self.x[0]) + \
                                           self.r2 * self.x[6] ** 2 * sym.cos(self.x[1]) + self.r1 * self.x[5] ** 2 * sym.cos(self.x[0]) - g) + \
                            self.r1 * FX_contact * sym.cos(self.x[0]) * sym.sin(self.x[0]) - sym.sin(self.x[0]) * (self.r1 * FY_contact * sym.sin(self.x[0]) - self.u[0]) + \
                            FK_extended * W * sym.cos(self.x[0])

        C_contact_extended = W * (self.M1 * self.r1 * self.x[5] ** 2 * sym.sin(self.x[0]) - FK_extended * sym.sin(self.x[0]) + FX_contact) - \
                            sym.cos(self.x[0]) * (FY_contact * self.r1 * sym.sin(self.x[0]) - FX_contact * self.r1 * sym.cos(self.x[0]) - self.u[0])

        D_contact_extended = W * (self.M1 * self.r1 * self.x[5] ** 2 * sym.cos(self.x[0]) - FK_extended * sym.cos(self.x[0]) + FY_contact - self.M1 * self.g) - \
                            sym.sin(self.x[0]) * (FY_contact * self.r1 * sym.sin(self.x[0]) - FX_contact * self.r1 * sym.cos(self.x[0]) - self.u[0])

        E_contact_extended = W * (FK_extended * self.r2 * sym.sin(self.x[1] - self.x[0]) + self.u[0]) - self.r2 * sym.cos(self.x[1] - self.x[0]) * \
                            (self.r1 * FY_contact * sym.sin(self.x[0]) - self.r1 * FX_contact * sym.cos(self.x[0]) - self.u[0])

        theta1_ddot_contact_extended = ((a4*b2/b4-a2)*E_contact_extended/e2-a3*C_contact_extended/c2+a4*b3*D_contact_extended/(b4*d2)+\
                                      A_contact_extended-a4*B_contact_extended/b4)/\
                                      (a1-b1*a4/b4-(a2-a4*b2/b4)*e1/e2-a3*c1/c2+a4*b3*d1/(b4*d2))

        theta2_ddot_contact_extended = (E_contact_extended-e1*theta1_ddot_contact_extended)/e2

        x_ddot_contact_extended = (C_contact_extended-c1*theta1_ddot_contact_extended)/c2

        y_ddot_contact_extended = (D_contact_extended-d1*theta1_ddot_contact_extended)/d2

        w_ddot_contact_extended = (A_contact_extended+B_contact_extended-(a1+b1)*theta1_ddot_contact_extended-(a2+b2)*theta2_ddot_contact_extended-\
                                  a3*x_ddot_contact_extended-b3*y_ddot_contact_extended)/(a4+b4)

        contact_extended_ddots = np.asarray([theta1_ddot_contact_extended, theta2_ddot_contact_extended, x_ddot_contact_extended, y_ddot_contact_extended, w_ddot_contact_extended])

        contact_extended_dynamics = np.hstack((self.x[5:], contact_extended_ddots))

        # contact with retracted leg
        contact_retracted_conditions = np.asarray([self.k0-self.x[4]+self.u[1]<=0,
                                                      self.x[3]-self.ground_height_function(self.x[2])<=0])
        A_contact_retracted = W * self.M2 * (self.x[5] ** 2 * W * sym.sin(self.x[0]) - 2 * self.x[5] * self.x[9] * sym.cos(self.x[0]) + \
                                           self.r2 * self.x[6] ** 2 * sym.sin(self.x[1]) + self.r1 * self.x[5] ** 2 * sym.sin(self.x[0])) - \
                            self.r1 * FX_contact * (sym.cos(self.x[0]) ** 2) + sym.cos(self.x[0]) * (self.r1 * FY_contact * sym.sin(self.x[0]) - self.u[0]) + \
                            FK_retracted * W * sym.sin(self.x[0])

        B_contact_retracted = W * self.M2 * (self.x[5] ** 2 * W * sym.cos(self.x[0]) + 2 * self.x[5] * self.x[9] * sym.sin(self.x[0]) + \
                                           self.r2 * self.x[6] ** 2 * sym.cos(self.x[1]) + self.r1 * self.x[5] ** 2 * sym.cos(self.x[0]) - g) + \
                            self.r1 * FX_contact * sym.cos(self.x[0]) * sym.sin(self.x[0]) - sym.sin(self.x[0]) * (self.r1 * FY_contact * sym.sin(self.x[0]) - self.u[0]) + \
                            FK_retracted * W * sym.cos(self.x[0])

        C_contact_retracted = W * (self.M1 * self.r1 * self.x[5] ** 2 * sym.sin(self.x[0]) - FK_retracted * sym.sin(self.x[0]) + FX_contact) - \
                            sym.cos(self.x[0]) * (FY_contact * self.r1 * sym.sin(self.x[0]) - FX_contact * self.r1 * sym.cos(self.x[0]) - self.u[0])

        D_contact_retracted = W * (self.M1 * self.r1 * self.x[5] ** 2 * sym.cos(self.x[0]) - FK_retracted * sym.cos(self.x[0]) + FY_contact - self.M1 * self.g) - \
                            sym.sin(self.x[0]) * (FY_contact * self.r1 * sym.sin(self.x[0]) - FX_contact * self.r1 * sym.cos(self.x[0]) - self.u[0])

        E_contact_retracted = W * (FK_retracted * self.r2 * sym.sin(self.x[1] - self.x[0]) + self.u[0]) - self.r2 * sym.cos(self.x[1] - self.x[0]) * \
                            (self.r1 * FY_contact * sym.sin(self.x[0]) - self.r1 * FX_contact * sym.cos(self.x[0]) - self.u[0])

        theta1_ddot_contact_retracted = ((a4*b2/b4-a2)*E_contact_retracted/e2-a3*C_contact_retracted/c2+a4*b3*D_contact_retracted/(b4*d2)+\
                                      A_contact_retracted-a4*B_contact_retracted/b4)/\
                                      (a1-b1*a4/b4-(a2-a4*b2/b4)*e1/e2-a3*c1/c2+a4*b3*d1/(b4*d2))

        theta2_ddot_contact_retracted = (E_contact_retracted-e1*theta1_ddot_contact_retracted)/e2

        x_ddot_contact_retracted = (C_contact_retracted-c1*theta1_ddot_contact_retracted)/c2

        y_ddot_contact_retracted = (D_contact_retracted-d1*theta1_ddot_contact_retracted)/d2

        w_ddot_contact_retracted = (A_contact_retracted+B_contact_retracted-(a1+b1)*theta1_ddot_contact_retracted-(a2+b2)*theta2_ddot_contact_retracted-\
                                  a3*x_ddot_contact_retracted-b3*y_ddot_contact_retracted)/(a4+b4)

        contact_retracted_ddots = np.asarray([theta1_ddot_contact_retracted, theta2_ddot_contact_retracted, x_ddot_contact_retracted, y_ddot_contact_retracted, w_ddot_contact_retracted])

        contact_retracted_dynamics = np.hstack((self.x[5:], contact_retracted_ddots))

        self.f_list = np.asarray([flight_extended_dynamics, flight_retracted_dynamics, contact_extended_dynamics, contact_retracted_dynamics])
        self.f_type_list = np.asarray(['continuous', 'continuous', 'continuous', 'continuous'])
        self.c_list = np.asarray([flight_extended_conditions, flight_retracted_conditions, contact_extended_conditions, contact_retracted_conditions])

        DTHybridSystem.__init__(self, self.f_list, self.f_type_list, self.x, self.u, self.c_list, \
                                self.initial_env)

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
        theta1 = self.env[self.x[0]]
        theta2 = self.env[self.x[1]]
        x0 = self.env[self.x[2]]
        y0 = self.env[self.x[3]]
        w = self.env[self.x[4]]
        theta1_dot = self.env[self.x[5]]
        theta2_dot = self.env[self.x[6]]
        x0_dot = self.env[self.x[7]]
        y0_dot = self.env[self.x[8]]
        w_dot =  self.env[self.x[9]]
        if not self.was_in_contact and y0-self.ground_height_function(x0)<=0:
            # just touched down
            # set the touchdown point
            self.env[self.xTD] = x0
        self.was_in_contact=y0-self.ground_height_function(x0)<=0
        #FIXME: _state_to_env does not set self.env[self.xTD]

    def _state_to_env(self, state, u=None):
        env = {}
        # print('state',state)
        for i, s_i in enumerate(state):
            env[self.x[i]] = s_i
        if u is None:
            for u_i in self.u:
                env[u_i] = 0
        else:
            for i, u_i in enumerate(u):
                env[self.u[i]] = u[i]
        # if touch down, set xTD
        if state[3]-self.ground_height_function(state[2])<=0:
            env[self.xTD] = state[2]
        return env
