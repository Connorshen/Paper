function plot_size_corr()
load("compare_size_corr.mat")
len = size(compare_size_corr,1);
corr_all = [];
cluster_sizes = [];
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
    corr_digits_all = [];
    for digit=digits
        ind = labels==digit;
        ind = indexs(ind);
        output_cpl_digit_all = output_cpl(ind,:);
        corr_digit_all=[];
        first_output_cpl_digit = output_cpl_digit_all(1,:);
        for j = 2:size(output_cpl_digit_all,1)
            current_output_cpl_digit = output_cpl_digit_all(j,:);
            corr_digit_all=[corr_digit_all;corr2(first_output_cpl_digit,current_output_cpl_digit)];
        end
        corr_digit = mean(corr_digit_all);
        corr_digits_all = [corr_digits_all;corr_digit];
    end
    corr_all = [corr_all;corr_digits_all'];
end
corr_all_mean = mean(corr_all,2);
corr_all_std = std(corr_all,0,2);
figure(1)
set(gcf,"Position",[500,500,600,400], "color","w")
errorbar(cluster_sizes,corr_all_mean,corr_all_std,'-b','LineWidth',1)
axis([1,40,0.3,0.7])
set(gca,'XTick',cluster_sizes);
set(gca,'xticklabel',cluster_sizes);
ylabel("cpl corr");
xlabel("cpl cluster size");
title("compare cpl cluster size");