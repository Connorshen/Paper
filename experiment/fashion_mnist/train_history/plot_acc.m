function plot_acc()
load("compare_acc.mat")
len = size(compare_acc,1);
rl_acc_all = [];
rl_loss_all = [];
for i = 1:len
    step_all = compare_acc{i,1}(:,1);
    step_index = step_all(compare_acc{i,1}(:,4)~=0);
    step_index = step_index(1:100);
    rl_acc = compare_acc{i,1}(:,4);
    rl_acc = rl_acc(step_index)';
    rl_acc_all = [rl_acc_all;rl_acc];
    rl_loss = compare_acc{i,1}(:,5);
    rl_loss = rl_loss(step_index)';
    rl_loss_all = [rl_loss_all;rl_loss];
end
rl_acc_all = mean(rl_acc_all,1);
rl_loss_all = mean(rl_loss_all,1);
figure(1)
set(gcf,'Position',[500,500,1200,500], 'color','w')
% fig normal rl acc
subplot(1,2,1);
plot(step_index,rl_acc_all,"b");
xlabel("step");
ylabel("acc");
title("普通强化学习acc")
axis([-inf,inf,0,1])
% fig normal rl loss
subplot(1,2,2);
plot(step_index,rl_loss_all,"b");
xlabel("step");
ylabel("loss");
title("普通强化学习loss")
axis([-inf,inf,0,1])