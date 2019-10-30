function plot_acc()
% batch verify_acc
load("batch_check_points");
step = check_points(:,1);
batch_verify_acc = check_points(:,4);
batch_verify_step = step(batch_verify_acc~=0);
batch_verify_acc = batch_verify_acc(batch_verify_acc~=0);
% batch verify_acc
load("rl_check_points");
step = check_points(:,1);
rl_verify_acc = check_points(:,4);
rl_verify_step = step(rl_verify_acc~=0);
rl_verify_acc = rl_verify_acc(rl_verify_acc~=0);
plot(rl_verify_step,rl_verify_acc,"b",batch_verify_step,batch_verify_acc,"r")
legend("rl verify acc","batch verify acc")
xlabel("step")
ylabel("acc")
title("数字0至4")
axis([-inf,inf,0,1])