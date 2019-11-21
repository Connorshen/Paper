clear
clc
% 无关的参数
data_ratio = 1;% 数据集比例
test_early_stopping = -1;% 测试的时候提早break的step，不想提早结束的话取-1
train_early_stopping = 1;% 训练的时候提早break的step，不想提早结束的话取-1
file_name = "train_history/compare_size_corr.mat";
compare_size_corr = cell(1,3);
% 初始化参数
digits = 0:9;
in_features_cpl = 2560;
out_features_cpl = 6000;
verify_step = 500;
get_lr_step = 100;
get_lr_batch = 100;
n_neuron_clusters = [2,5,10,15,30];
% 加载数据集
data = load_mnist_data(digits,data_ratio);
disp("load data success");
for i=1:length(n_neuron_clusters)
    n_neuron_cluster = n_neuron_clusters(i);
    init_para = init_paramter(digits,in_features_cpl,out_features_cpl,n_neuron_cluster,verify_step,get_lr_step,get_lr_batch);
    disp("init para success")
    init_para
    rand('state',1);
    net = init_net(init_para);
    disp("init net success")
    [rl_check_points,rl_best_train_result] = rl_trainer(init_para,net,data,train_early_stopping,test_early_stopping);
    compare_size_corr{i,1} = rl_check_points;
    compare_size_corr{i,2} = rl_best_train_result.init_para.n_neuron_cluster;
    compare_size_corr{i,3} = rl_best_train_result;
end
save(file_name,"compare_size_corr","-v7.3");
plot_size_corr()