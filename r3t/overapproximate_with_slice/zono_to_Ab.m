function [A_obs, b_obs] = zono_to_Ab(obs_zono, Z)
    obs_dim = [1; 2]; % note that the obstacle exists in the x-y space (not theta or v)
    buffer_dist = 0;

    obstacle = obs_zono.Z;
    c = Z(obs_dim, 1);
    G = Z(:, 2:end);
    k_no_slc_G = G(obs_dim, :);


    buff_obstacle_c = [obstacle(:, 1) - c];
    buff_obstacle_G = [obstacle(:, 2:end), k_no_slc_G, buffer_dist*eye(2)]; % obstacle is "buffered" by non-k-sliceable part of FRS
    buff_obstacle_G(:, ~any(buff_obstacle_G)) = []; % delete zero columns of G
    buff_obstacle = [buff_obstacle_c, buff_obstacle_G];
    [A_obs, b_obs] = polytope_PH(buff_obstacle);
end