function plot_out_features()
load("compare_cluster_size.mat")
len = size(compare_cluster_size,1);
trial = size(compare_cluster_size,2)/2;
cluster_sizes = [];
acc_final_mean = [];
acc_final_std = [];
acc_all_mean = [];
acc_all_std =[];
for i = 1:len
    step_all = compare_cluster_size{i,1}(:,1);
    step_index = step_all(compare_cluster_size{i,1}(:,4)~=0);
    cluster_size = compare_cluster_size{i,2};
    cluster_sizes = [cluster_sizes;cluster_size];
    acc_max = [];
    acc_all = [];
    for j = 1:trial
        rl_acc = compare_cluster_size{i,j*2-1}(:,4);
        rl_acc = rl_acc(step_index)';
        acc_all = [acc_all;rl_acc];
        acc_max = [acc_max;max(rl_acc)];
    end
    acc_final_mean = [acc_final_mean;mean(acc_max)];
    acc_final_std = [acc_final_std;std(acc_max)];
    acc_all_mean = [acc_all_mean;mean(acc_all)];
    acc_all_std = [acc_all_std;std(acc_all)];
end
figure(1)
set(gcf,"Position",[500,500,1200,1200], "color","w")
subplot(2,2,2);
errorbar(1:size(cluster_sizes,1),acc_final_mean,acc_final_std,'-b','LineWidth',1)
axis([0,6,0.6,1])
set(gca,'XTick',1:size(cluster_sizes,1));
set(gca,'xticklabel',cluster_sizes);
ylabel("acc");
xlabel("cpl cluster size");
title("compare cpl cluster size");
subplot(2,2,1);
colors = ['k','c','m','r','b'];
for i=1:len
    acc_mean = acc_all_mean(i,:);
    acc_std = acc_all_std(i,:);
    color = colors(i);
    shadedErrorBar(step_index,acc_mean,acc_std,"lineprops",color);
    hold on;
end
legend(num2str(cluster_sizes),"Location","SouthEast");
ylabel("acc");
xlabel("step");
title("compare cpl cluster size");