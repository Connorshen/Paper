function plot_size_self_corr()
close all
load("compare_size_corr.mat")
len = size(compare_size_corr,1);
cluster_sizes = [];
cos_sim_all = [];
for i=1:len
    net = compare_size_corr{i,3}.net;
    init_para = compare_size_corr{i,3}.init_para;
    n_neuron_cluster = init_para.n_neuron_cluster;
    cluster_sizes = [cluster_sizes;n_neuron_cluster];
    data_ratio = 0.1;% 数据集比例
    early_stopping = 10;% 测试的时候提早break的step，不想提早结束的话取-1
    data = load_mnist_data(init_para.digits,data_ratio);
    [predict_result]=run_predicting(net,init_para,data,early_stopping);
    output_cpl = predict_result.output_cpl;
    labels = predict_result.label;
    indexs = 1:size(labels,1);
    digits = init_para.digits;
    cos_sim_digit = [];
    for digit=digits
        ind = labels==digit;
        ind = indexs(ind);
        output_cpl_digit_all = output_cpl(ind,:);
        y = [];
        for j=1:size(output_cpl_digit_all,1)
            x = output_cpl_digit_all(j,:);
            y = [y;x/norm(x)];
        end
        cos_sim_n = (svds(y,1)^2-1)/(size(y,1)-1);
        cos_sim_digit = [cos_sim_digit;cos_sim_n];
    end
    cos_sim_all = [cos_sim_all;cos_sim_digit'];
end
figure(1)
fig_para = fig_paramter();
plot(1:size(cluster_sizes,1),cos_sim_all,"LineWidth",fig_para.linewidth)
axis([0,6,0,1])
set(gca,'XTick',1:size(cluster_sizes,1));
set(gca,'XTickLabel',cluster_sizes);
set(gca, "FontSize", fig_para.fontsize);
ylabel("Correlation coefficient","FontSize", fig_para.fontsize);
xlabel("Cluster size of CPL","FontSize", fig_para.fontsize);
title("Correlation coefficient of intra-class");
legend(num2str((0:9)'),"Location","NorthEast");

set(gcf, "PaperUnits", "inches");
set(gcf, "PaperSize", [6 4]);
set(gcf, "PaperPositionMode", "manual");
set(gcf, "PaperPosition", [0 0 6 4]);

print(gcf, "-dpng", "CorrelationCoefficientIntraClass.png");