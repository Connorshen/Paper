function plot_out_features()
load("compare_cluster_size.mat")
len = size(compare_cluster_size,1);
trial = size(compare_cluster_size,2)/2;
cluster_sizes = [];
acc_final_mean = [];
acc_final_std = [];
acc_all_mean = [];
loss_all_mean = [];
acc_all_std =[];
for i = 1:len
    step_all = compare_cluster_size{i,1}(:,1);
    step_index = step_all(compare_cluster_size{i,1}(:,4)~=0);
    cluster_size = compare_cluster_size{i,2};
    cluster_sizes = [cluster_sizes;cluster_size];
    acc_max = [];
    acc_all = [];
    loss_all = [];
    for j = 1:trial
        rl_acc = compare_cluster_size{i,j*2-1}(:,4);
        rl_acc = rl_acc(step_index)';
        rl_loss = compare_cluster_size{i,j*2-1}(:,5);
        rl_loss = rl_loss(step_index)';
        acc_all = [acc_all;rl_acc];
        loss_all = [loss_all;rl_loss];
        acc_max = [acc_max;max(rl_acc)];
    end
    acc_final_mean = [acc_final_mean;mean(acc_max)];
    acc_final_std = [acc_final_std;std(acc_max)];
    acc_all_mean = [acc_all_mean;mean(acc_all)];
    acc_all_std = [acc_all_std;std(acc_all)];
    loss_all_mean =[loss_all_mean;mean(loss_all)];
end
figure(1)
fig_para = fig_paramter();
subplot(2,2,1);
colors = ['k','c','m','r','b'];
for i=1:len
    acc_mean = acc_all_mean(i,:);
    acc_std = acc_all_std(i,:);
    color = colors(i);
    s = shadedErrorBar(step_index,acc_mean,acc_std,"LineProps",color);
    set(s.mainLine,"LineWidth",fig_para.linewidth)
    hold on;
end
legend(num2str(cluster_sizes),"Location","SouthEast");
ylabel("Accuracy","FontSize", fig_para.fontsize);
xlabel("Step","FontSize", fig_para.fontsize);
set(gca, "FontSize", fig_para.fontsize);
title("Training process of different cluster sizes");
subplot(2,2,2);
plot(step_index,loss_all_mean,"LineWidth",fig_para.linewidth)
ylabel("Cross entropy loss","FontSize", fig_para.fontsize);
xlabel("Step","FontSize", fig_para.fontsize);
set(gca, "FontSize", fig_para.fontsize);
title("Training process of different cluster sizes");
legend(num2str(cluster_sizes),"Location","NorthEast");
subplot(2,2,[3 4]);
errorbar(1:size(cluster_sizes,1),acc_final_mean,acc_final_std,"-b","LineWidth",fig_para.linewidth)
axis([0,6,0.6,1])
set(gca,"XTick",1:size(cluster_sizes,1));
set(gca,"XTickLabel",cluster_sizes);
set(gca, "FontSize", fig_para.fontsize);
ylabel("Accuracy","FontSize", fig_para.fontsize);
xlabel("Cluster size of CPL","FontSize", fig_para.fontsize);
title("Accuracy of different cluster sizes");


set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [10 8]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 8]);

print(gcf, '-depsc2', 'ImpactOfClusterSizeOnAccuracy.eps');
