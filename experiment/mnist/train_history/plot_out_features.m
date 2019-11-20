function plot_out_features()
load("compare_out_features.mat")
len = size(compare_out_features,1);
trial = size(compare_out_features,2)/2;
out_features_cpls = [];
rl_acc_all_mean = [];
rl_acc_all_std = [];
for i = 1:len
    step_all = compare_out_features{i,1}(:,1);
    step_index = step_all(compare_out_features{i,1}(:,4)~=0);
    out_features_cpl = compare_out_features{i,2};
    out_features_cpls = [out_features_cpls;out_features_cpl];
    rl_acc_trial = [];
    for j = 1:trial
        rl_acc = compare_out_features{i,j*2-1}(:,4);
        rl_acc = rl_acc(step_index)';
        rl_acc = max(rl_acc);
        rl_acc_trial = [rl_acc_trial;rl_acc];
    end
    rl_acc_all_mean = [rl_acc_all_mean;mean(rl_acc_trial)];
    rl_acc_all_std = [rl_acc_all_std;std(rl_acc_trial)];
end
figure(1)
set(gcf,"Position",[500,500,1200,500], "color","w")
errorbar(out_features_cpls,rl_acc_all_mean,rl_acc_all_std,'-b','LineWidth',1)  
axis([5000,100000,0,1])
set(gca,'XTick',out_features_cpls);
set(gca,'xticklabel',out_features_cpls);
ylabel("acc");
xlabel("cpl out features");
title("compare cpl out features");