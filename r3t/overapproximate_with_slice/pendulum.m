%% Step1: This example illustrate  composition of parameterized trajecotry
% can do interseting things:
% the pendulum equation is \ddot{\theta} = -g/L\sin(\theta) - \alpha \theta
% the states are y1 = \theta, y2 = \dot{\theta}
figure(1);clf;
% Initial condition
y0 = [0 0];
  
tspan = [0 2];
y_all = y0;
t_all = 0;

for i = 1:10
   k = mod(i,2)*2-1; %osslilate desired speed
[t,y] = ode45(@(t,y)damped_pendulum(t,y,k,y_all(end,:)), tspan, y_all(end,:));
t_all = [t_all; t_all(end)+t(2:end)];
y_all = [y_all; y(2:end,:)];

end
plot(y_all(:,1),y_all(:,2))
% legend('theta','theta dot')
%

%% Step 2: Now we can come up with the symbolic representation of the closed loop dynamics
% Note saturation can be implemented but the non-linearity is going to
% cause problems
L = 1.0;    % lentth of pendulum
g = 9.81;    % gravitational acceleration
alpha = 0.6; % damping factor
K = 10;
saturation_limit = 5;
    
syms theta theta_dot theta0 theta_dot0 k t
dyn = [theta; theta_dot; theta0; theta_dot0; k; t]
% controller
tau = K *(k-theta_dot); %saturation can be easily done by limiting the range of k
% tau = sat(tau,saturation_limit);
theta_dotdot = ( -g/L*sin(theta) - alpha*theta_dot + tau );
       
ddyn = [theta_dot; theta_dotdot;  0;  0; 0; 1];
syms tdummy udummy
matlabFunction(ddyn, 'File', 'dyn_pendulum_syms', 'vars', {tdummy dyn udummy});    
%save dynamics

%%  Step 3: Use CORA and the dynamics to generate Forward reachable set for a interval
% We arbitarily set T = 0.3 s. Initial condition: theta = [-0.5, 0.5] theta_dot = [-0.5, 0.5]
% desired velocity k = [0,0.2]
dim = 6;
t_final = 0.3;
% set options for reachability analysis:
options.taylorTerms=15; % number of taylor terms for reachable sets
options.zonotopeOrder= 100; % zonotope order... increase this for more complicated systems.
options.maxError = 1000*ones(dim, 1); % our zonotopes shouldn't be "splitting", so this term doesn't matter for now
options.verbose = 1;
options.uTrans = 0; % we won't be using any inputs, as traj. params specify trajectories
options.U = zonotope([0, 0]);
options.advancedLinErrorComp = 0;
options.tensorOrder = 1;
options.reductionInterval = inf;
options.reductionTechnique = 'girard';

options.tStart = 0;
options.tFinal = t_final;
options.timeStep = 0.01; 
 
x0 = zeros(dim,1); x0(5) = 0.1;
options.x0 = x0;
gen_theta = zeros(dim,1);gen_theta(1) = 0.5;gen_theta(3) = 0.5;
gen_theta_dot = zeros(dim,1);gen_theta_dot(2) = 0.5;gen_theta_dot(4) = 0.5;
gen_k = zeros(dim,1);gen_k(5) = 0.1;
% gen_Au = zeros(14,1);gen_Au(11) = 0;

options.R0 = zonotope( [options.x0 gen_theta gen_theta_dot gen_k] );
sys = nonlinearSys(dim, 1, @dyn_pendulum_syms, options );
tic
vehRS = reach(sys, options);
toc
%% Now online we can 'slice' the generated FRS given a random initial condition and desired velocity
% We verify the correctness and conservativeness with a ode45 simulation 
% Notice this takes a short time if you comment out the plotting;
% ruun the following two sections multiple times to see different randon
% slices.
clc;
figure(3);clf;hold on;
simk = randRange(0,0.2);
simtheta = randRange(-0.5,0.5);
simthetadot = randRange(-0.5,0.5);
tic
for i = 1:length(vehRS)
    p_all = plotFilled(vehRS{i}{1}, [1, 2], 'g'); %this is the full set, it is very big 
    p_all.FaceAlpha = .05;
    p_all.EdgeAlpha = 0.1;
    
    zonoslice = zonotope_slice(vehRS{i}{1}, [3;4;5], [simtheta; simthetadot; simk]); % this is the subset
%     zonoslice = deleteAligned(vehRS{i}{1});
    
    p_FRS = plotFilled(zonoslice, [1, 2], 'r');
    p_FRS.FaceAlpha = .3;
    p_FRS.EdgeAlpha = 0.2;
    
end
slicing_time_with_plotting = toc
[t,y]=ode45(@(t,y)damped_pendulum(t,y,simk,[simtheta;simthetadot]),[0 t_final], [simtheta;simthetadot]);
plot(y(:,1),y(:,2));
%% Obstalces intersection can also be done with as polytope constraints
% obstacle can appear in any dimension in state space, here just assume a
% obstacle at a theta = 0.2 for all velocities
tic
obs_dim = [1 2];
k_dim = 5;
A_obs_array= []; b_obs_array = [];
eps = 0.01;
obs_zono = zonotope([[0.2;0] [0.05;0] [0;0.5]]);
plot(obs_zono,[1 2],'r');

for t_idx = 1:length(vehRS)
     
zono_one = vehRS{i}{1};

zono_one = zonotope_slice(zono_one, [3;4;5],  [simtheta; simthetadot; simk]);

Z = zono_one.Z; 
Zc = center(zono_one);
%consider each obstacle as a halfspace constraint


[A_obs, b_obs]=zono_to_Ab(obs_zono,Z);
if  max(A_obs * [0;0]-b_obs) <= eps
    display("!!UNSAFE!!")
    break;
end
end
collision_check_time_with_plotting = toc
save_FRS = cell(0);
info_FRS = cell(0);
slice_dim = [3,4,5];
for i = 1:length(vehRS)
   save_FRS{end+1} = get(vehRS{i}{1}, 'Z'); 
   G = save_FRS{end}(:, 2:end);
   slice_idx = [];
   for j = 1:length(slice_dim)
       myidxs = find(G(slice_dim(j), :) ~= 0);
       if length(myidxs) ~= 1
           if length(myidxs) == 0
               error('No generator for slice index');
           else
               error('More than one generator for slice index');
           end
       end
       slice_idx(j, 1) = myidxs;
    end
   info_FRS{end+1} = slice_idx-1;%since pyhon starts with 0
end
save('test_zono.mat','save_FRS','info_FRS');
%% Helper
function x_sat = sat(x,lim)
x_sat = (2/(1+exp(-2*x/lim))-1)*lim;
end