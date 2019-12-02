function plot_inhibition_threshold()
close all
load("compare_inhibition_threshold.mat")
n_trial = size(compare_inhibition_threshold,1);
n_threshold = (size(compare_inhibition_threshold,2)-2)/2;
thresholds = [];
acc_normal_all = [];
acc_normal_std = [];
acc_inhibition_all = [];
acc_inhibition_std = [];
for i =1:n_threshold
    acc_normal_trial = [];
    acc_inhibition_trial = [];
    for j=1:n_trial
        init_para = compare_inhibition_threshold{j,i*2};
        step_all = compare_inhibition_threshold{j,i*2-1}(:,1);
        step_index = step_all(compare_inhibition_threshold{j,i*2-1}(:,4)~=0);
        acc = compare_inhibition_threshold{j,i*2-1}(:,4);
        acc = acc(step_index);
        acc = max(acc);
        acc_inhibition_trial = [acc_inhibition_trial;acc];
        if j==1
            thresholds = [thresholds;init_para.inhibition_threshold];
        end
        if i==n_threshold
            acc = compare_inhibition_threshold{j,i*2+1}(:,4);
            acc = acc(step_index);
            acc = max(acc);
            acc_normal_trial = [acc_normal_trial;acc];
        end
    end
    acc_inhibition_all = [acc_inhibition_all;mean(acc_inhibition_trial)];
    acc_inhibition_std = [acc_inhibition_std;std(acc_inhibition_trial)];
    if size(acc_normal_trial,1)~=0
        for j=1:n_threshold
            acc_normal_all = [acc_normal_all;mean(acc_normal_trial)];
            acc_normal_std = [acc_normal_std;std(acc_normal_trial)];
        end
    end
end
figure(1)
fig_para = fig_paramter();
plot(1:size(thresholds,1),acc_normal_all,"LineWidth",fig_para.linewidth);
hold on;
errorbar(1:size(thresholds,1),acc_inhibition_all,acc_inhibition_std,"LineWidth",fig_para.linewidth);
axis([0.5,5.5,0,1])
xlabel("Threshold","FontSize", fig_para.fontsize)
ylabel("Accuracy","FontSize", fig_para.fontsize)
set(gca,"XTick",1:size(thresholds,1));
set(gca,"XTickLabel",thresholds);
set(gca, "FontSize", fig_para.fontsize);
legend(["normal","inhibition"],"Location","NorthEast");
title("Inhibition CPL with threshold")
set(gcf, "PaperUnits", "inches");
set(gcf, "PaperSize", [6 4]);
set(gcf, "PaperPositionMode", "manual");
set(gcf, "PaperPosition", [0 0 6 4]);
set(gca, "FontWeight", fig_para.fontweight);
print(gcf, "-dpng", "InhibitionCPLWithThresholds.png");