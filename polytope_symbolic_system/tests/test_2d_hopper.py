from examples.hopper_2d import Hopper_2d
import matplotlib.pyplot as plt
import numpy as np
import math
def test_freefall_hopper():
    initial_state = np.asarray([0.5, 5, 0, 0, 2, 0., 0, 0., 0., 0.])
    hopper = Hopper_2d(initial_state=initial_state)
    # simulate the hopper
    state_count = 30000
    states = np.zeros([10, state_count])
    states[:, 0] = initial_state
    cg_coordinate_states = np.zeros([10, state_count])
    end_count = None
    step_size = 1e-4
    i = 0
    try:
        for i in range(1,state_count):
            if i%10000==0:
                print('iteration %i' %i)
            # foot length controller
            states[:, i] = hopper.forward_step(step_size=step_size, u=np.asarray([0, 0]))
            # print(states[:,i])
            if i%100==0:
                print(hopper.env[hopper.xTD], hopper.env[hopper.x[1]], hopper.was_in_contact)
    except Exception as e:
        print("Simulation terminated due to exception: %s" %e)
        end_count = i
        print('Simulation terminated at step %i' %i)
        print('Env is:', hopper.get_current_state())
    # plot the Raibert coordinate states
    fig1, ax1 = plt.subplots(5,1)
    labels1 = ['$x_{ft}$', '$y_{ft}$', '$\\theta$', '$\\phi$', 'r']
    for i in range(5):
        ax1[i].plot(states[i,0:end_count])
        ax1[i].set_xlabel('Steps')
        ax1[i].set_ylabel(labels1[i])
    ax1[1].set_ylim([0,6])
    ax1[4].set_ylim([0,6])

    fig3, ax3 = plt.subplots(5,1)
    labels3 = ['$\\dot{x_{ft}}$', '$\\dot{y_{ft}}$', '$\\dot{\\theta}$', '$\\dot{\\phi}$', '$\\dot{r}$']
    for i in range(5):
        ax3[i].plot(states[i+5,1:end_count])
        ax3[i].set_xlabel('Steps')
        ax3[i].set_ylabel(labels3[i])
    ax3[1].set_ylim([-6,6])
    ax3[4].set_ylim([-6,6])

    plt.show()

def test_pushoff_hopper():
    initial_state = np.asarray([0.5, 1, 0, 0, 5, 0, 0., 0., 0., 0.])
    hopper = Hopper_2d(initial_state=initial_state)
    # simulate the hopper
    state_count = 1000
    states = np.zeros([10, state_count])
    states[:, 0] = initial_state
    end_count = None
    step_size = 1e-2
    i = 0
    try:
        for i in range(1,state_count):
            if i%10000==0:
                print('iteration %i' %i)
            # foot length controller
            # if states[1, i-1]<=0 and states[4,i-1]>0 and states[4,i-1]<15:
            #     f = 500
            # elif states[4,i-1]<0 or states[4,i-1]>15:
            #     f = -100*(states[4,i-1]-initial_state[4])-1*states[9,i-1]
            # ascending
            if states[9,i-1]>=0:
                u = np.asarray([0,120])#300
            else:
                u = np.asarray([0, 60])  # 300
            states[:, i] = hopper.forward_step(step_size=step_size, u=u)
            # print(states[:,i])
            if i%100==0 or hopper.was_in_contact:
                print(hopper.env[hopper.xTD]-hopper.env[hopper.x[0]], hopper.env[hopper.x[1]], hopper.was_in_contact)
    except Exception as e:
        print("Simulation terminated due to exception: %s" %e)
        end_count = i
        print('Simulation terminated at step %i' %i)
        print('Env is:', hopper.get_current_state())
    # plot the Raibert coordinate states
    fig1, ax1 = plt.subplots(5,1)
    labels1 = ['$x_{ft}$', '$y_{ft}$', '$\\theta$', '$\\phi$', 'r']
    for i in range(5):
        ax1[i].plot(states[i,0:end_count])
        ax1[i].set_xlabel('Steps')
        ax1[i].set_ylabel(labels1[i])
    ax1[1].set_ylim([0,2])
    ax1[4].set_ylim([0,2])

    fig3, ax3 = plt.subplots(5,1)
    labels3 = ['$\\dot{x_{ft}}$', '$\\dot{y_{ft}}$', '$\\dot{\\theta}$', '$\\dot{\\phi}$', '$\\dot{r}$']
    for i in range(5):
        ax3[i].plot(states[i+5,1:end_count])
        ax3[i].set_xlabel('Steps')
        ax3[i].set_ylabel(labels3[i])
    ax3[1].set_ylim([-6,6])
    ax3[4].set_ylim([-6,6])

    plt.show()

def test_moving_hopper():
    initial_state = np.asarray([0.5, 1, 0, 0, 5, 0.5, 0., 0., 0., 0.])
    hopper = Hopper_2d(initial_state=initial_state)
    # simulate the hopper
    state_count = 1000
    states = np.zeros([10, state_count])
    states[:, 0] = initial_state
    end_count = None
    step_size = 1e-2
    i = 0
    try:
        for i in range(1,state_count):
            if i%10000==0:
                print('iteration %i' %i)
            # foot length controller
            f = 0
            alpha_des = -np.arctan(states[5,i-1]/states[6,i-1])
            print(alpha_des)
            if states[1, i-1]<=0 and states[4,i-1]>0 and states[4,i-1]<15:
                f = (states[3, i-1]-alpha_des)
            # elif states[4,i-1]<0 or states[4,i-1]>15:
            #     f = -100*(states[4,i-1]-initial_state[4])-1*states[9,i-1]
            # ascending
            if states[9,i-1]>=0:
                u = np.asarray([f,120])#300
            else:
                u = np.asarray([f, 60])  # 300
            states[:, i] = hopper.forward_step(step_size=step_size, u=u)
            # print(states[:,i])
            if i%100==0 or hopper.was_in_contact:
                print(hopper.env[hopper.xTD]-hopper.env[hopper.x[0]], hopper.env[hopper.x[1]], hopper.was_in_contact)
    except Exception as e:
        print("Simulation terminated due to exception: %s" %e)
        end_count = i
        print('Simulation terminated at step %i' %i)
        print('Env is:', hopper.get_current_state())
    # plot the Raibert coordinate states
    fig1, ax1 = plt.subplots(5,1)
    labels1 = ['$x_{ft}$', '$y_{ft}$', '$\\theta$', '$\\phi$', 'r']
    for i in range(5):
        ax1[i].plot(states[i,0:end_count])
        ax1[i].set_xlabel('Steps')
        ax1[i].set_ylabel(labels1[i])
    ax1[1].set_ylim([0,2])
    ax1[4].set_ylim([0,2])

    fig3, ax3 = plt.subplots(5,1)
    labels3 = ['$\\dot{x_{ft}}$', '$\\dot{y_{ft}}$', '$\\dot{\\theta}$', '$\\dot{\\phi}$', '$\\dot{r}$']
    for i in range(5):
        ax3[i].plot(states[i+5,1:end_count])
        ax3[i].set_xlabel('Steps')
        ax3[i].set_ylabel(labels3[i])
    ax3[1].set_ylim([-6,6])
    ax3[4].set_ylim([-6,6])

    plt.show()

def test_raibert_controller_hopper(desired_lateral_velocity=0.0):
    initial_state = np.asarray([0.,0.,15.,1.,0.8,0.,0.,0.,0.,0.])
    hopper = Hopper_2d(initial_state=initial_state)
    # simulate the hopper
    state_count = 300000
    states = np.zeros([10, state_count])
    states[:, 0] = initial_state
    cg_coordinate_states = np.zeros([10, state_count])
    end_count = None
    step_size = 1e-4
    wasInContact = False
    desiredAlpha = 0.
    try:
        for i in range(1,state_count):
            if i%10000==0:
                print('iteration %i' %i)
            # foot length controller
            chi = 1.0
            in_contact = states[3, i - 1] <= 0
            x, y, theta, alpha, l, x_dot, y_dot, theta_dot, alpha_dot, l_dot = hopper.get_cg_coordinate_states()
            if in_contact:
                if y_dot>0:
                    chi = 1.15
            # thigh torque controller

            kp = 1.0  # proportional gain in flight
            kd = 1.0  # derivative gain in flight
            output = 0  # control output
            # Stance phase
            kp2 = .5  # proportional gain in stance
            kd2 = .5  # derivative gain in stance
            kvel = .2  # offset for acceleration
            if in_contact:
                wasInContact = True  # system has reached stance phase
                output = kp2 * (theta) + kd2 * theta_dot  # PD control corrects for theta to make body upright
            # Flight
            else:
                if wasInContact:  # system just left stance
                    desiredAlpha = -math.atan2(x_dot, y_dot)  # want to land with alpha pointing at CG footprint
                # PD control over footprint, plus a small displacement for acceleration/deceleration
                output = -kp * (alpha - desiredAlpha) - kd * alpha_dot - kvel * (x_dot - desired_lateral_velocity)
                wasInContact = False  # system is no longer in stance
            tau = output

            states[:, i] = hopper.forward_step(step_size=step_size, u=np.asarray([tau, chi]))
            cg_coordinate_states[:, i] = hopper.get_cg_coordinate_states()
    except Exception as e:
        print("Simulation terminated due to exception: %s" %e)
        end_count = i
        print('Simulation terminated at step %i' %i)
        print('Env is:', hopper.get_current_state())
    # plot the Raibert coordinate states
    fig1, ax1 = plt.subplots(5,1)
    labels1 = ['$\\theta_1$', '$\\theta_2$', '$x_0$', '$y_0$', 'w']
    for i in range(5):
        ax1[i].plot(states[i,1:end_count])
        ax1[i].set_xlabel('Steps')
        ax1[i].set_ylabel(labels1[i])
    ax1[0].set_ylim([-np.pi, np.pi])
    ax1[1].set_ylim([-np.pi, np.pi])
    ax1[2].set_ylim([-2+initial_state[2],3+initial_state[2]])
    ax1[3].set_ylim([-0.5,3])
    ax1[4].set_ylim([0, 3])

    fig3, ax3 = plt.subplots(5,1)
    labels3 = ['$\\dot{\\theta_1}$', '$\\dot{\\theta_2}$', '$\\dot{x_0}$', '$\\dot{y_0}$', '\\dot{w}']
    for i in range(5):
        ax3[i].plot(states[i+5,1:end_count])
        ax3[i].set_xlabel('Steps')
        ax3[i].set_ylabel(labels3[i])
    ax3[0].set_ylim([-np.pi, np.pi])
    ax3[1].set_ylim([-np.pi, np.pi])
    ax3[2].set_ylim([-2,3])
    ax3[3].set_ylim([-0.5,3])
    ax3[4].set_ylim([0, 3])

    # plot the cg coordinate states
    fig2, ax2 = plt.subplots(5,1)
    labels2 = ['$x$', '$y$', '$\\theta$', '$\\alpha$', 'l']
    for i in range(5):
        ax2[i].plot(cg_coordinate_states[i,1:end_count])
        ax2[i].set_xlabel('Steps')
        ax2[i].set_ylabel(labels2[i])
    ax2[0].set_ylim([-2+initial_state[2],3+initial_state[2]])
    ax2[1].set_ylim([-0.5,4])
    ax2[2].set_ylim([-np.pi, np.pi])
    ax2[3].set_ylim([-np.pi, np.pi])
    ax2[4].set_ylim([0, 3])
    plt.show()

if __name__=='__main__':
    test_moving_hopper()