import pydrake.symbolic as sym
from common.symbolic_system import *

def test_with_toy_system():
    x = np.array([sym.Variable('x')])
    u = np.array([sym.Variable('u')])
    parameters = np.array([sym.Variable('param_'+str(i)) for i in range(2)])
    f = np.atleast_2d([sym.pow(x[0],3)+sym.sin(u[0])])
    print('f:', f)
    dyn = ContinuousDynamics(f,x,u)
    print('A:', dyn.A)
    print('B:', dyn.B)
    print('c:', dyn.c)
    env = {x[0]:3,u[0]:4}
    print('Nonlinear x_dot:', dyn.evaluate_xdot(env))
    print('linearized x_dot:', dyn.evaluate_xdot(env, linearize=True))

    sys = DTContinuousSystem(f, x, u, initial_env={x[0]:4, u[0]:0})
    print('Nonlinear forward step:', sys.forward_step(u=np.array([2]), step_size=1, modify_system=False))
    print('Linear forward step:', sys.forward_step(u=np.array([2]), linearlize=True, step_size = 1, modify_system=False))
    print(sys.get_reachable_polytopes([0.5]))
if __name__ == '__main__':
    test_with_toy_system()
