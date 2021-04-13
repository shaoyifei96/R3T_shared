function dydt = damped_pendulum(t,y,k,y0)

    % original equation: m*L^2\ddot{theta} = tau - mgL\sin(\theta) - \alpha\dot{\theta}
    % tau is torque
    % the pendulum equation is \ddot{\theta} = tau/m/L^2 -g/L\sin(\theta) - \alpha/m/L^2 \dot{\theta}
    % the states are y1 = \theta, y2 = \dot{\theta}
   
    % for code simplicity taking tau = tau/m/L^2 and alpha = \alpha/m/L^2
    
    %k = spd_change +-1
    %
    L = 1.0;    % lentth of pendulum
    g = 9.81;    % gravitational acceleration
    alpha = 0.6; % damping factor
    K = 10;
    saturation_limit = 5;
    % Without external torque
%     dydt = [y(2); 
%            ( -g/L*sin(y(1)) - alpha*y(2) )];
%        
    % With external torque
            %ref - actual
    tau = K *(k-y(2));
    
%     if abs(tau)>saturation_limit
%         tau = sign(tau)*saturation_limit;
%     end
    dydt = [y(2); 
           ( -g/L*sin(y(1)) - alpha*y(2) + tau )];
    
end


