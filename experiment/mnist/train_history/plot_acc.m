function plot_acc()
load("compare_acc.mat")
len = size(compare_convergence,1);
rl_acc_all = [];
batch_acc_all = [];
for i = 1:len
    step_all = compare_convergence{i,1}(:,1);
    step_index = step_all(compare_convergence{i,1}(:,4)~=0);
    rl_acc = compare_convergence{i,1}(:,4);
    rl_acc = rl_acc(step_index)';
    rl_acc_all = [rl_acc_all;rl_acc];
    batch_acc = compare_convergence{i,2}(:,4);
    batch_acc = batch_acc(step_index)';
    batch_acc_all = [batch_acc_all;batch_acc];
end
rl_acc_all = mean(rl_acc_all,1);
batch_acc_all = mean(batch_acc_all,1);
figure(1)
% fig normal rl
subplot(1,2,1);
plot(step_index,rl_acc_all,"r");
xlabel("step");
ylabel("acc");
title("普通强化学习")
axis([-inf,inf,0,1])
% fig batch rl
subplot(1,2,2);
plot(step_index,batch_acc_all,"r");
xlabel("step");
ylabel("acc");
title("batch强化学习")
axis([-inf,inf,0,1])