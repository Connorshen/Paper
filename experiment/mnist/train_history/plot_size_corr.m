function plot_size_corr()
load("compare_size_corr.mat")
len = size(compare_size_corr,1);
for i=1:len
    net = compare_size_corr{i,3}.net;
    init_para = compare_size_corr{i,3}.init_para;
    n_neuron_cluster = init_para.n_neuron_cluster;
    data_ratio = 0.1;% 数据集比例
    early_stopping = 1;% 测试的时候提早break的step，不想提早结束的话取-1
    data = load_mnist_data(init_para.digits,data_ratio);
    [predict_result]=run_predicting(net,init_para,data,early_stopping);
    input_cpl = predict_result.input_cpl;
    output_cpl = predict_result.output_cpl;
    img_origin = predict_result.img_origin;
    labels = predict_result.label;
    indexs = 1:size(labels,1);
    digits = init_para.digits;
    for digit=digits
        ind = labels==digit;
        ind = indexs(ind);
    end
end