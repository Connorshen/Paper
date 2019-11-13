function plot_cluster_size()
load("compare_cluster_size.mat")
len = size(compare_cluster_size,1);
trial = size(compare_cluster_size,2)/2;
n_neuron_clusters = [];
rl_acc_all = [];
for i = 1:len
    step_all = compare_cluster_size{i,1}(:,1);
    step_index = step_all(compare_cluster_size{i,1}(:,4)~=0);
    n_neuron_cluster = compare_cluster_size{i,2};
    n_neuron_clusters = [n_neuron_clusters;n_neuron_cluster];
    rl_acc_trial = [];
    for j = 1:trial
        rl_acc = compare_cluster_size{i,j*2-1}(:,4);
        rl_acc = rl_acc(step_index)';
        rl_acc_trial = [rl_acc_trial;rl_acc];
    end
    rl_acc_trial = mean(rl_acc_trial,1);
    rl_acc_all = [rl_acc_all;rl_acc_trial];
end
rl_acc_all = max(rl_acc_all')';
figure(1)
set(gcf,'Position',[500,500,600,400], 'color','w')
bar(rl_acc_all);
for i=1:len
    text(i,rl_acc_all(i),num2str(rl_acc_all(i)),'VerticalAlignment','bottom','HorizontalAlignment','center');
end
set(gca,'xticklabel',n_neuron_clusters);
axis([-inf,inf,0.75,1])
ylabel("acc");
xlabel("cluster size");
title("compare cluster size");