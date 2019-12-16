clear
clc
% 无关的参数
trial = 3;
data_ratio = 0.1;% 数据集比例
test_early_stopping = -1;% 测试的时候提早break的step，不想提早结束的话取-1
train_early_stopping = 1000;% 训练的时候提早break的step，不想提早结束的话取-1
file_name = "train_history/compare_final_acc.mat";
compare_final_acc = cell(trial,2);
% 初始化参数
digits = 0:1;
in_features_cpl = 2560;
out_features_cpl = 5000;
verify_step = 500;
get_lr_step = 10;
get_lr_batch = 100;
n_neuron_cluster = 10;
init_para = init_paramter(digits,in_features_cpl,out_features_cpl,n_neuron_cluster,verify_step,get_lr_step,get_lr_batch);
disp("init para success")
init_para
% 加载数据集
data = load_mnist_data(init_para.digits,data_ratio);
disp("load data success");
for i = 1:trial
    rand('state',i);
    % 训练普通rl
    net = init_net(init_para,data.train_img,data.train_label);
    disp("init net success")
    [rl_check_points,rl_best_train_result] = rl_trainer(init_para,net,data,train_early_stopping,test_early_stopping);
    compare_final_acc{i,1} = rl_check_points;
end
compare_final_acc{i,2} = rl_best_train_result;
save(file_name,"compare_final_acc","-v7.3");