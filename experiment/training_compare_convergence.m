clear
clc
% 无关的参数
data_ratio = 0.1;% 数据集比例
test_early_stopping = 10;% 测试的时候提早break的step，不想提早结束的话取-1
rl_file_name = "rl_convergence.mat";
batch_rl_file_name = "batch_rl_convergence.mat";
% 初始化参数
digits = 0:1;
in_features_cpl = 2560;
out_features_cpl = 5000;
verify_step = 100;
get_lr_step = 100;
get_lr_batch = 40;
init_para = init_paramter(digits,in_features_cpl,out_features_cpl,verify_step,get_lr_step,get_lr_batch);
disp("init para success")
init_para
% 加载数据集
data = load_mnist_data(init_para.digits,data_ratio);
disp("load data success");
% 训练普通rl
net = init_net(init_para);
disp("init net success")
[rl_check_points,rl_best_train_result] = rl_trainer(init_para,net,data,test_early_stopping);
save(strcat("train_history/",rl_file_name),"rl_check_points");
save(strcat("net_weight/",rl_file_name),"rl_best_train_result");
% 训练rl_batch
net = init_net(init_para);
disp("init net success")
[rl_batch_check_points,rl_batch_best_train_result] = rl_batch_trainer(init_para,net,data,test_early_stopping);
save(strcat("train_history/",batch_rl_file_name),"rl_batch_check_points");
save(strcat("net_weight/",batch_rl_file_name),"rl_batch_best_train_result");