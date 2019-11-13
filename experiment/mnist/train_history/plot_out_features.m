function plot_out_features()
load("compare_out_features.mat")
len = size(compare_out_features,1);
trial = size(compare_out_features,2)/2;
out_features_cpls = [];
rl_acc_all = [];
for i = 1:len
    step_all = compare_out_features{i,1}(:,1);
    step_index = step_all(compare_out_features{i,1}(:,4)~=0);
    out_features_cpl = compare_out_features{i,2};
    out_features_cpls = [out_features_cpls;out_features_cpl];
    rl_acc_trial = [];
    for j = 1:trial
        rl_acc = compare_out_features{i,j*2-1}(:,4);
        rl_acc = rl_acc(step_index)';
        rl_acc_trial = [rl_acc_trial;rl_acc];
    end
    rl_acc_trial = mean(rl_acc_trial,1);
    rl_acc_all = [rl_acc_all;rl_acc_trial];
end
rl_acc_all_max = max(rl_acc_all')';
figure(1)
subplot(1,2,1);
plot(step_index,rl_acc_all);
hold on;
plot(step_index,0.1*ones(size(step_index)),"black--");
ylabel("acc");
xlabel("step");
title("compare cpl out features");
legend([num2str(out_features_cpls);"random"],"Location","NorthWest");
subplot(1,2,2);
set(gcf,"Position",[500,500,1200,500], "color","w")
bar(rl_acc_all_max);
hold on;
plot(0.1*ones(len),"black--");
legend(["final acc","random"],"Location","NorthWest");
for i=1:len
    text(i,rl_acc_all_max(i),num2str(rl_acc_all_max(i)),"VerticalAlignment","bottom","HorizontalAlignment","center");
end
set(gca,"xticklabel",out_features_cpls);
ylabel("acc");
xlabel("cpl out features");
title("compare cpl out features");