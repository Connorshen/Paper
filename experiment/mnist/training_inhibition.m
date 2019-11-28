clear
clc
% 无关的参数
trial = 2;
data_ratio = 1;% 数据集比例
test_early_stopping = 10;% 测试的时候提早break的step，不想提早结束的话取-1
train_early_stopping = 1000;% 训练的时候提早break的step，不想提早结束的话取-1
file_name = "train_history/compare_inhibition.mat";
compare_inhibition = cell(1,1);
% 初始化参数
digits = 0:4;
inhibition_activity = false;
inhibition_threshold = 0.3;
in_features_cpl = 2560;
out_features_cpl = 500;
verify_step = 100;
get_lr_step = 10;
get_lr_batch = 100;
n_neuron_cluster = 10;
init_para = init_paramter(digits,in_features_cpl,out_features_cpl,n_neuron_cluster,verify_step,get_lr_step,get_lr_batch);
init_para.inhibition_threshold = inhibition_threshold;
disp("init para success")
init_para
% 加载数据集
data = load_mnist_data(init_para.digits,data_ratio);
disp("load data success");
% 开始训练
for i = 1:trial
    rand('state',1);
    % 训练普通rl
    init_para.inhibition_activity = false;
    net = init_net(init_para);
    disp("init net success")
    [rl_check_points,rl_best_train_result] = rl_trainer(init_para,net,data,train_early_stopping,test_early_stopping);
    compare_inhibition{i,1} = rl_check_points;
    compare_inhibition{i,2} = rl_best_train_result.init_para;
    % 训练抑制rl
    init_para.inhibition_activity = true;
    net = init_net(init_para);
    disp("init net success")
    [rl_check_points,rl_best_train_result] = rl_trainer(init_para,net,data,train_early_stopping,test_early_stopping);
    compare_inhibition{i,3} = rl_check_points;
    compare_inhibition{i,4} = rl_best_train_result.init_para;
end
save(file_name,"compare_inhibition","-v7.3");