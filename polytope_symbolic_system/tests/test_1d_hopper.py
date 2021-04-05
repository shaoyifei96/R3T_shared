from examples.hopper_1d import *
import matplotlib.pyplot as plt

def test_static_hopper():
    initial_state = np.asarray([2, 0])
    hopper = Hopper_1d(f_max=0, initial_state=initial_state)
    #simulate the hopper
    state_count = 25000
    states = np.zeros([2, state_count])
    states[:,0] = initial_state
    for i in range(state_count):
        states[:,i] = hopper.forward_step(step_size=1e-3)
    plt.subplot(211)
    plt.plot(states[0,:])
    plt.xlabel('Steps')
    plt.ylabel('$x$')
    plt.subplot(212)
    plt.plot(states[1,:])
    plt.xlabel('Steps')
    plt.ylabel('$\dot{x}$')
    plt.show()

def test_constant_input_hopper():
    initial_state = np.asarray([2, 0])
    f_max = 5
    hopper = Hopper_1d(f_max=f_max, initial_state=initial_state)
    # simulate the hopper
    state_count = 25000
    states = np.zeros([2, state_count])
    states[:, 0] = initial_state
    for i in range(state_count):
        u = f_max
        states[:, i] = hopper.forward_step(step_size=1e-3, u=np.asarray([u]))
    plt.subplot(211)
    plt.plot(states[0, :])
    plt.xlabel('Steps')
    plt.ylabel('$x$')
    plt.subplot(212)
    plt.plot(states[1, :])
    plt.xlabel('Steps')
    plt.ylabel('$\dot{x}$')
    plt.show()

def test_bang_bang_hopper():
    initial_state = np.asarray([2.,0.])
    l = 1
    p = 0.1
    step_size = 0.04
    hopper_system = Hopper_1d(l=l, p=p, initial_state= initial_state, f_max=25)
    goal_state = np.asarray([3,0.0])
    goal_tolerance = 2e-2


if __name__=='__main__':
    test_constant_input_hopper()