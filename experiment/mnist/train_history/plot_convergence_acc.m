function plot_convergence_acc()
close all
load("compare_convergence.mat")
len = size(compare_convergence,1);
rl_acc_all = [];
batch_acc_all = [];
rl_loss_all = [];
batch_loss_all = [];
step_all = compare_convergence{1,1}(:,1);
step_index = step_all(compare_convergence{1,1}(:,5)~=0);
for i = 1:1
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
fig_para = fig_paramter();
subplot(1,2,1);
plot(step_index,rl_acc_all_mean,"LineWidth",fig_para.linewidth);
hold on;
plot(step_index,batch_acc_all_mean,"LineWidth",fig_para.linewidth);
xlabel("Step","FontSize", fig_para.fontsize);
ylabel("Accuracy","FontSize", fig_para.fontsize);
set(gca, "FontSize", fig_para.fontsize);
legend("rl","rl batch","Location","SouthEast")
axis([0,inf,0,1])
title("Convergence speed of accuracy")
subplot(1,2,2);
plot(step_index,rl_loss_all,"Color",fig_para.colors(1,:),"LineWidth",fig_para.linewidth);
hold on;
plot(step_index,batch_loss_all,"Color",fig_para.colors(2,:),"LineWidth",fig_para.linewidth);
legend("rl","rl batch");
xlabel("Step","FontSize", fig_para.fontsize);
ylabel("Cross entropy loss","FontSize", fig_para.fontsize);
set(gca, "FontSize", fig_para.fontsize);
axis([-inf,inf,0,1])
title("Convergence speed of loss")

set(gcf, "PaperUnits", "inches");
set(gcf, "PaperSize", [12 4]);
set(gcf, "PaperPositionMode", "manual");
set(gcf, "PaperPosition", [0 0 12 4]);

print(gcf, "-dpng", "ConvergenceSpeedOfTwoAlgorithm.png");