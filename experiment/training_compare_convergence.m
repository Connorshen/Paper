clear
clc
% 无关的参数
trial = 3;
data_ratio = 0.1;% 数据集比例
test_early_stopping = 100;% 测试的时候提早break的step，不想提早结束的话取-1
train_early_stopping = 300;% 训练的时候提早break的step，不想提早结束的话取-1
file_name = "train_history/compare_convergence.mat";
compare_convergence = cell(trial,4);% [rl_check_points,rl_best_train_result,rl_batch_check_points,rl_batch_best_train_result]
% 初始化参数
digits = 0:5;
in_features_cpl = 2560;
out_features_cpl = 10000;
verify_step = 20;
get_lr_step = 10;
get_lr_batch = 100;
init_para = init_paramter(digits,in_features_cpl,out_features_cpl,verify_step,get_lr_step,get_lr_batch);
disp("init para success")
init_para
% 加载数据集
data = load_mnist_data(init_para.digits,data_ratio);
disp("load data success");
for i = 1:trial
    rand('state',i);
    % 训练普通rl
    net = init_net(init_para);
    disp("init net success")
    [rl_check_points,rl_best_train_result] = rl_trainer(init_para,net,data,train_early_stopping,test_early_stopping);
    compare_convergence{i,1} = rl_check_points;
    compare_convergence{i,2} = rl_best_train_result;
    % 训练rl_batch
    net = init_net(init_para);
    disp("init net success")
    [rl_batch_check_points,rl_batch_best_train_result] = rl_batch_trainer(init_para,net,data,train_early_stopping,test_early_stopping);
    compare_convergence{i,3} = rl_batch_check_points;
    compare_convergence{i,4} = rl_batch_best_train_result;
end
save(file_name,"compare_convergence","-v7.3");
plot_convergence_acc()
