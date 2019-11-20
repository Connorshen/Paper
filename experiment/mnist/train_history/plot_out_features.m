function plot_out_features()
load("compare_out_features.mat")
len = size(compare_out_features,1);
trial = size(compare_out_features,2)/2;
out_features_cpls = [];
acc_final_mean = [];
acc_final_std = [];
acc_all_mean = [];
acc_all_std =[];
for i = 1:len
    step_all = compare_out_features{i,1}(:,1);
    step_index = step_all(compare_out_features{i,1}(:,4)~=0);
    out_features_cpl = compare_out_features{i,2};
    out_features_cpls = [out_features_cpls;out_features_cpl];
    acc_max = [];
    acc_all = [];
    for j = 1:trial
        rl_acc = compare_out_features{i,j*2-1}(:,4);
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
set(gcf,"Position",[500,500,1200,500], "color","w")
subplot(1,2,2);
errorbar(out_features_cpls,acc_final_mean,acc_final_std,'-b','LineWidth',1)
axis([5000,100000,0.3,1])
set(gca,'XTick',out_features_cpls);
set(gca,'xticklabel',out_features_cpls);
ylabel("acc");
xlabel("cpl out features");
title("compare cpl out features");
subplot(1,2,1);
colors = ['k','c','m','r','b'];
for i=1:len
    acc_mean = acc_all_mean(i,:);
    acc_std = acc_all_std(i,:);
    color = colors(i);
    shadedErrorBar(step_index,acc_mean,acc_std,"lineprops",color);
    hold on;
end
legend(num2str(out_features_cpls),"Location","SouthEast");
ylabel("acc");
xlabel("step");
title("compare cpl out features");