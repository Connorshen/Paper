clear
clc
% 无关的参数
rand('state',0);
data_ratio = 0.1;% 数据集比例
test_early_stopping = 10;% 测试的时候提早break的step，不想提早结束的话取-1
train_early_stopping = 200;% 训练的时候提早break的step，不想提早结束的话取-1
rl_file_name = "rl_convergence.mat";
batch_rl_file_name = "rl_batch_convergence.mat";
% 初始化参数
digits = 0:2;
in_features_cpl = 2560;
out_features_cpl = 10000;
verify_step = 10;
get_lr_step = 100;
get_lr_batch = 100;
init_para = init_paramter(digits,in_features_cpl,out_features_cpl,verify_step,get_lr_step,get_lr_batch);
disp("init para success")
init_para
% 加载数据集
data = load_mnist_data(init_para.digits,data_ratio);
disp("load data success");
% 训练普通rl
net = init_net(init_para);
disp("init net success")
[rl_check_points,rl_best_train_result] = rl_trainer(init_para,net,data,train_early_stopping,test_early_stopping);
save(strcat("train_history/t_",rl_file_name),"rl_check_points");
save(strcat("net_weight/w_",rl_file_name),"rl_best_train_result","-v7.3");
clear rl_check_points rl_best_train_result net
% 训练rl_batch
net = init_net(init_para);
disp("init net success")
[rl_batch_check_points,rl_batch_best_train_result] = rl_batch_trainer(init_para,net,data,train_early_stopping,test_early_stopping);
save(strcat("train_history/t_",batch_rl_file_name),"rl_batch_check_points");
save(strcat("net_weight/w_",batch_rl_file_name),"rl_batch_best_train_result","-v7.3");
plot_convergence_acc()