function plot_convergence_acc()

% rl_verify_acc
load("t_rl_convergence.mat");
rl_step = rl_check_points(:,1);
rl_verify_acc = rl_check_points(:,4);
rl_verify_step = rl_step(rl_verify_acc~=0);
rl_verify_acc = rl_verify_acc(rl_verify_acc~=0);
% rl_batch_verify_acc
load("t_rl_batch_convergence.mat");
rl_batch_step = rl_batch_check_points(:,1);
rl_batch_verify_acc = rl_batch_check_points(:,4);
rl_batch_verify_step = rl_batch_step(rl_batch_verify_acc~=0);
rl_batch_verify_acc = rl_batch_verify_acc(rl_batch_verify_acc~=0);
plot(rl_verify_step,rl_verify_acc,'r',rl_batch_verify_step,rl_batch_verify_acc,"b")
legend("rl verify acc","rl batch verify acc")
axis([-inf,inf,0,1])