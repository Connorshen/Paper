clear
clc
% 无关的参数
data_ratio = 0.1;% 数据集比例
test_early_stopping = 10;% 测试的时候提早break的step，不想提早结束的话取-1
rl_history_file_name = "check_points_rl_convergence.mat";
rl_weight_file_name = "train_result_rl_convergence.mat";
% 初始化参数
init_para = init_paramter();
init_para.verify_step = 100;
init_para.out_features_cpl = 5000;
init_para.n_neuron_cluster = 10;
disp("init para success")
init_para
net = init_net(init_para);
disp("init net success")
[rl_check_points,rl_best_train_result] = rl_trainer(init_para,net,data_ratio,test_early_stopping);
save(strcat("train_history/",rl_history_file_name),"rl_check_points");
save(strcat("net_weight/",rl_weight_file_name),"rl_best_train_result");