from examples.pendulum import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = "Times New Roman"
matplotlib.rcParams.update({'font.size': 16})

def test_static_pendulum():
    pend = Pendulum(initial_state = np.array([np.pi/2,0]), b=0.2)
    #simulate the pendulum
    state_count = 4000
    states = np.zeros([2, state_count])
    states[:,0] = np.array([np.pi/2,0])
    for i in range(state_count):
        states[:,i] = pend.forward_step(step_size=1e-2)
    plt.subplot(211)
    plt.plot(states[0,:])
    plt.xlabel('Steps')
    plt.ylabel('$\\theta$')
    plt.subplot(212)
    plt.plot(states[1,:])
    plt.xlabel('Steps')
    plt.ylabel('$\dot{\\theta}$')
    plt.show()

def test_controlled_pendulum():
    pend = Pendulum(initial_state = np.array([np.pi/2,0]), b=0.2,input_limits=np.array([[-.98],[.98]]))
    #simulate the pendulum for 200 steps
    state_count = 4000
    states = np.zeros([2, state_count])
    states[:,0] = np.array([np.pi/2.2,0])
    #Use PID controller to drive the pendulum upright
    for i in range(state_count):
        current_state = pend.get_current_state()
        u = 10*(current_state[0]-np.pi)+1*(current_state[1])
        states[:,i] = pend.forward_step(step_size=1e-2, u=np.asarray([-u]))
    plt.subplot(211)
    plt.plot(states[0,:])
    plt.xlabel('Steps')
    plt.ylabel('$\\theta$')
    plt.subplot(212)
    plt.plot(states[1,:])
    plt.xlabel('Steps')
    plt.ylabel('$\dot{\\theta}$')
    plt.show()

def test_bang_bang_pendulum():
    m = 1
    l = 0.5
    g = 9.8
    b = 0.1
    pend = Pendulum(initial_state= np.array([0,1e-3]), input_limits=np.asarray([[-1],[1]]), m=m, l=l, g=g, b=b)
    #simulate the pendulum for 200 steps
    state_count = 10000
    states = np.zeros([2, state_count])
    states[:,0] = np.array([0,0])
    #Use PID controller to drive the pendulum upright
    completed_index = state_count
    for i in range(state_count):
        current_state = pend.get_current_state()
        u_comp = b*current_state[1]
        # print(u_comp)
        if current_state[1]>0:
            u = min(1, 1+u_comp)
        elif current_state[1]<0:
            u = max(-1, -1+u_comp)
        else:
            u = u_comp
        err =current_state[0]-np.pi
        if abs(err)<0.5:
            u = -1*err+u_comp
        states[:,i] = pend.forward_step(step_size=1e-3, u=np.asarray([u]))
        if states[0,i] >= np.pi or states[0,i] <= -np.pi:
            completed_index=i
            break
    # plt.subplot(211)
    # plt.plot(states[0,:completed_index+1])
    # plt.xlabel('Steps')
    # plt.ylabel('$\\theta$')
    # plt.subplot(212)
    # plt.plot(states[1,:completed_index+1])
    # plt.xlabel('Steps')
    # plt.ylabel('$\dot{\\theta}$')
    # plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(states[0,:completed_index+1], states[1,:completed_index+1], 'cyan','-')
    ax.scatter([0],[0], facecolor='red', s=10)
    ax.scatter([np.pi], [0], facecolor='green', s=10)
    ax.scatter([-np.pi], [0], facecolor='green', s=10)
    ax.grid(True, which='both')
    y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(y_formatter)

    ax.set_yticks(np.arange(-10,10,2))
    ax.set_xlim([-4,4])
    ax.set_ylim([-10,10])
    plt.tight_layout(h_pad=-0.5)
    ax.set_xlabel('$\\theta (rad)$')
    ax.set_ylabel('$\dot{\\theta} (rad/s)$')
    ax.set_title('Pendulum Swing-up Optimal Trajectory')
    # plt.show()
    plt.gcf().subplots_adjust(left=0.13, bottom=0.12)
    plt.savefig('pendulum_bang_bang_path.png', dpi=500)
if __name__ == '__main__':
    test_bang_bang_pendulum()