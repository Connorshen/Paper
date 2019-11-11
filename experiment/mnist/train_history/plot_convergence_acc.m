function plot_convergence_acc()
load("compare_convergence.mat")
len = size(compare_convergence,1);
step_index_acc = [300,350,400,450,500];
rl_acc_all = [];
batch_acc_all = [];
rl_loss_all = [];
batch_loss_all = [];
step_all = compare_convergence{1,1}(:,1);
step_index_loss = step_all(compare_convergence{1,1}(:,5)~=0);
for i = 1:len
    rl_acc = compare_convergence{i,1}(:,4);
    rl_acc = rl_acc(step_index_acc)';
    rl_acc_all = [rl_acc_all;rl_acc];
    batch_acc = compare_convergence{i,2}(:,4);
    batch_acc = batch_acc(step_index_acc)';
    batch_acc_all = [batch_acc_all;batch_acc];
    rl_loss = compare_convergence{i,1}(:,5);
    rl_loss = rl_loss(step_index_loss)';
    rl_loss_all = [rl_loss_all;rl_loss];
    batch_loss = compare_convergence{i,2}(:,5);
    batch_loss = batch_loss(step_index_loss)';
    batch_loss_all = [batch_loss_all;batch_loss];
end
rl_acc_all = mean(rl_acc_all,1);
batch_acc_all = mean(batch_acc_all,1);
acc = [rl_acc_all',batch_acc_all'];
rl_loss_all = mean(rl_loss_all,1);
batch_loss_all = mean(batch_loss_all,1);
figure(1)
set(gcf,'Position',[500,500,1200,500], 'color','w')
subplot(1,2,1);
bar(step_index_acc,acc);
xlabel("step");
ylabel("acc");
legend('rl','rl batch')
axis([-inf,inf,0.8,1])
title("acc compare")
subplot(1,2,2);
plot(step_index_loss,rl_loss_all,"r",step_index_loss,batch_loss_all,"b");
legend("rl","rl batch");
xlabel("step");
ylabel("cross entropy loss ");
axis([-inf,inf,0,1])
title("loss compare")