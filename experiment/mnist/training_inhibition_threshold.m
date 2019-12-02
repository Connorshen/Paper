clear
clc
% 无关的参数
trial = 5;
data_ratio = 1;% 数据集比例
test_early_stopping = -1;% 测试的时候提早break的step，不想提早结束的话取-1
train_early_stopping = 10000;% 训练的时候提早break的step，不想提早结束的话取-1
file_name = "train_history/compare_inhibition_threshold.mat";
compare_inhibition_threshold = cell(1,1);
% 初始化参数
inhibition_thresholds = 0.1:0.2:0.9;
digits = 0:9;
in_features_cpl = 2560;
out_features_cpl = 1000;
verify_step = 500;
get_lr_step = 10;
get_lr_batch = 100;
n_neuron_cluster = 10;
% 加载数据集
data = load_mnist_data(digits,data_ratio);
disp("load data success");
% 开始训练
for i=1:length(inhibition_thresholds)
    
    inhibition_threshold = inhibition_thresholds(i);
    
    init_para = init_paramter(digits,in_features_cpl,out_features_cpl,n_neuron_cluster,verify_step,get_lr_step,get_lr_batch);
    init_para.inhibition_threshold = inhibition_threshold;
    for j = 1:trial
        rand('state',j);
        % 训练普通rl
        init_para.inhibition_activity = true;
        init_para
        net = init_net(init_para);
        disp("init net success")
        [rl_check_points,rl_best_train_result] = rl_trainer(init_para,net,data,train_early_stopping,test_early_stopping);
        compare_inhibition_threshold{j,i*2-1} = rl_check_points;
        compare_inhibition_threshold{j,i*2} = rl_best_train_result.init_para;
        % 训练抑制rl
        if i == length(inhibition_thresholds)
            init_para.inhibition_activity = false;
            init_para
            net = init_net(init_para);
            disp("init net success")
            [rl_check_points,rl_best_train_result] = rl_trainer(init_para,net,data,train_early_stopping,test_early_stopping);
            compare_inhibition_threshold{j,i*2+1} = rl_check_points;
            compare_inhibition_threshold{j,i*2+2} = rl_best_train_result.init_para;
        end
    end
end
save(file_name,"compare_inhibition_threshold","-v7.3");
plot_inhibition_threshold()