function plot_class()
close all
load("compare_class.mat")
n_classes = size(compare_class,1);
n_out_features = size(compare_class,2)/2;
n_categoryies = [];
out_features_cpls = [];
acc_all = [];
for i=1:n_classes
    acc_class = [];
    for j=1:n_out_features
        init_para = compare_class{i,j*2};
        out_features_cpl = init_para.out_features_cpl;
        n_category = init_para.n_category;
        step_all = compare_class{i,j*2-1}(:,1);
        step_index = step_all(compare_class{i,j*2-1}(:,4)~=0);
        acc = compare_class{i,j*2-1}(:,4);
        acc = acc(step_index);
        acc = max(acc);
        acc_class = [acc_class;acc];
        if i==1
            out_features_cpls = [out_features_cpls;out_features_cpl];
        end
    end
    n_categoryies = [n_categoryies;n_category];
    acc_all = [acc_all;acc_class'];
end
figure(1)
fig_para = fig_paramter();
plot(1:size(n_categoryies,1),acc_all,"LineWidth",fig_para.linewidth);
axis([0.5,5.5,0,1.1])
set(gca,"XTick",1:size(n_categoryies,1));
set(gca,"XTickLabel",n_categoryies);
set(gca, "FontSize", fig_para.fontsize);
xlabel("Number of categories","FontSize", fig_para.fontsize)
ylabel("Accuracy","FontSize", fig_para.fontsize)
legend(num2str(out_features_cpls),"Location","SouthWest");
title("CPL scale with number of categories")
set(gcf, "PaperUnits", "inches");
set(gcf, "PaperSize", [6 4]);
set(gcf, "PaperPositionMode", "manual");
set(gcf, "PaperPosition", [0 0 6 4]);
set(gca, "FontWeight", fig_para.fontweight);
print(gcf, "-depsc2", "CPLScaleWithCategories.eps");