function plot_inhibition()
close all
load("compare_inhibition.mat")
n_trial = size(compare_inhibition,1);
n_class = size(compare_inhibition,2)/4;
n_categoryies = [];
acc_normal_all = [];
acc_normal_std = [];
acc_inhibition_all = [];
acc_inhibition_std = [];
for i =1:n_class
    acc_normal_trial = [];
    acc_inhibition_trial = [];
    for j=1:n_trial
        init_para = compare_inhibition{j,i*4-2};
        step_all = compare_inhibition{j,i*4-3}(:,1);
        step_index = step_all(compare_inhibition{j,i*4-3}(:,4)~=0);
        acc = compare_inhibition{j,i*4-3}(:,4);
        acc = acc(step_index);
        acc = max(acc);
        acc_normal_trial = [acc_normal_trial;acc];
        acc = compare_inhibition{j,i*4-1}(:,4);
        acc = acc(step_index);
        acc = max(acc);
        acc_inhibition_trial = [acc_inhibition_trial;acc];
        if j == 1
            n_categoryies = [n_categoryies;init_para.n_category];
        end 
    end
    acc_normal_all = [acc_normal_all;mean(acc_normal_trial)];
    acc_normal_std = [acc_normal_std;std(acc_normal_trial)];
    acc_inhibition_all = [acc_inhibition_all;mean(acc_inhibition_trial)];
    acc_inhibition_std = [acc_inhibition_std;std(acc_inhibition_trial)];
end
figure(1)
fig_para = fig_paramter();
errorbar(1:size(n_categoryies,1),acc_normal_all,acc_normal_std,"LineWidth",fig_para.linewidth);
hold on;
errorbar(1:size(n_categoryies,1),acc_inhibition_all,acc_inhibition_std,"LineWidth",fig_para.linewidth);
axis([0.5,5.5,0,1.1])
xlabel("Number of categories","FontSize", fig_para.fontsize)
ylabel("Accuracy","FontSize", fig_para.fontsize)
legend(["normal","inhibition"],"Location","NorthEast");
set(gca,"XTick",1:size(n_categoryies,1));
set(gca,"XTickLabel",n_categoryies);
set(gca, "FontSize", fig_para.fontsize);
title("Inhibition CPL with number of categories")
set(gcf, "PaperUnits", "inches");
set(gcf, "PaperSize", [6 4]);
set(gcf, "PaperPositionMode", "manual");
set(gcf, "PaperPosition", [0 0 6 4]);
set(gca, "FontWeight", fig_para.fontweight);
print(gcf, "-dpng", "InhibitionCPLWithCategories.png");