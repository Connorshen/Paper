function plot_out_features()
close all
load("compare_out_features.mat")
len = size(compare_out_features,1);
trial = size(compare_out_features,2)/2;
out_features_cpls = [];
acc_final_mean = [];
acc_final_std = [];
acc_all_mean = [];
loss_all_mean = [];
acc_all_std =[];
for i = 1:len
    step_all = compare_out_features{i,1}(:,1);
    step_index = step_all(compare_out_features{i,1}(:,4)~=0);
    out_features_cpl = compare_out_features{i,2};
    out_features_cpls = [out_features_cpls;out_features_cpl];
    acc_max = [];
    acc_all = [];
    loss_all = [];
    for j = 1:trial
        rl_acc = compare_out_features{i,j*2-1}(:,4);
        rl_acc = rl_acc(step_index)';
        rl_loss = compare_out_features{i,j*2-1}(:,5);
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
    s = shadedErrorBar(step_index,acc_mean,acc_std,"lineprops",color);
    set(s.mainLine,"LineWidth",fig_para.linewidth)
    hold on;
end
legend(num2str(out_features_cpls),"Location","SouthEast");
ylabel("Accuracy","FontSize", fig_para.fontsize);
xlabel("Step","FontSize", fig_para.fontsize);
set(gca, "FontSize", fig_para.fontsize);
title("Training process of different CPL scales");
subplot(2,2,2);
plot(step_index,loss_all_mean,"LineWidth",fig_para.linewidth)
ylabel("Cross entropy loss","FontSize", fig_para.fontsize);
xlabel("Step","FontSize", fig_para.fontsize);
set(gca, "FontSize", fig_para.fontsize);
title("Training process of different CPL scales");
legend(num2str(out_features_cpls),"Location","NorthEast");
subplot(2,2,[3 4]);
errorbar(out_features_cpls,acc_final_mean,acc_final_std,"-b","LineWidth",fig_para.linewidth)
axis([5000,100000,0.3,1])
set(gca,"XTick",out_features_cpls);
set(gca,"XTickLabel",out_features_cpls);
set(gca, "FontSize", fig_para.fontsize);
ylabel("Accuracy","FontSize", fig_para.fontsize);
xlabel("Scales of CPL","FontSize", fig_para.fontsize);
title("Accuracy of different CPL scales");

set(gcf, "PaperUnits", "inches");
set(gcf, "PaperSize", [12 8]);
set(gcf, "PaperPositionMode", "manual");
set(gcf, "PaperPosition", [0 0 12 8]);

print(gcf, "-dpng", "ImpactOfCPLScaleOnAccuracy.png");
