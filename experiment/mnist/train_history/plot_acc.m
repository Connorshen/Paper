function plot_acc()
load("compare_acc.mat")
len = size(compare_convergence,1);
rl_acc_all = [];
batch_acc_all = [];
rl_loss_all = [];
batch_loss_all = [];
for i = 1:len
    step_all = compare_convergence{i,1}(:,1);
    step_index = step_all(compare_convergence{i,1}(:,4)~=0);
    rl_acc = compare_convergence{i,1}(:,4);
    rl_acc = rl_acc(step_index)';
    rl_acc_all = [rl_acc_all;rl_acc];
    batch_acc = compare_convergence{i,2}(:,4);
    batch_acc = batch_acc(step_index)';
    batch_acc_all = [batch_acc_all;batch_acc];
    rl_loss = compare_convergence{i,1}(:,5);
    rl_loss = rl_loss(step_index)';
    rl_loss_all = [rl_loss_all;rl_loss];
    batch_loss = compare_convergence{i,2}(:,5);
    batch_loss = batch_loss(step_index)';
    batch_loss_all = [batch_loss_all;batch_loss];
end
rl_acc_all = mean(rl_acc_all,1);
batch_acc_all = mean(batch_acc_all,1);
rl_loss_all = mean(rl_loss_all,1);
batch_loss_all = mean(batch_loss_all,1);
figure(1)
% fig normal rl acc
subplot(2,2,1);
plot(step_index,rl_acc_all,"r");
xlabel("step");
ylabel("acc");
title("普通强化学习acc")
axis([-inf,inf,0,1])
% fig batch rl acc
subplot(2,2,2);
plot(step_index,batch_acc_all,"r");
xlabel("step");
ylabel("acc");
title("batch强化学习acc")
axis([-inf,inf,0,1])
% fig normal rl loss
subplot(2,2,3);
plot(step_index,rl_loss_all,"r");
xlabel("step");
ylabel("loss");
title("普通强化学习loss")
axis([-inf,inf,0,1])
% fig batch rl loss
subplot(2,2,4);
plot(step_index,batch_loss_all,"r");
xlabel("step");
ylabel("loss");
title("batch强化学习loss")
axis([-inf,inf,0,1])