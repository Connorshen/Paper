function plot_convergence_acc()
load("compare_convergence.mat")
len = size(compare_convergence,1);
rl_acc_all = [];
batch_acc_all = [];
rl_loss_all = [];
batch_loss_all = [];
step_all = compare_convergence{1,1}(:,1);
step_index = step_all(compare_convergence{1,1}(:,5)~=0);
for i = 1:len
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
rl_acc_all_mean = mean(rl_acc_all,1);
rl_acc_all_std = std(rl_acc_all,1);
batch_acc_all_mean = mean(batch_acc_all,1);
batch_acc_all_std = std(batch_acc_all,1);
rl_loss_all = mean(rl_loss_all,1);
batch_loss_all = mean(batch_loss_all,1);
figure(1)
set(gcf,'Position',[500,500,1200,400], 'color','w')
subplot(1,2,1);
shadedErrorBar(step_index,rl_acc_all_mean,rl_acc_all_std,"lineprops",'r')
hold on;
shadedErrorBar(step_index,batch_acc_all_mean,batch_acc_all_std,"lineprops",'b')
xlabel("step");
ylabel("acc");
legend('rl','rl batch')
axis([-inf,inf,0,1])
title("acc compare")
subplot(1,2,2);
plot(step_index,rl_loss_all,"r",step_index,batch_loss_all,"b");
legend("rl","rl batch");
xlabel("step");
ylabel("cross entropy loss ");
axis([-inf,inf,0,1])
title("loss compare")