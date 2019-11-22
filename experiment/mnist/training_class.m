clear
clc
% 无关的参数
data_ratio = 1;% 数据集比例
test_early_stopping = -1;% 测试的时候提早break的step，不想提早结束的话取-1
train_early_stopping = 10000;% 训练的时候提早break的step，不想提早结束的话取-1
file_name = "train_history/compare_class.mat";
compare_class = cell(1,1);
% 初始化参数
in_features_cpl = 2560;
out_features_cpls = 10000:20000:100000;
n_classes = [2,4,6,8,10];
verify_step = 500;
get_lr_step = 100;
get_lr_batch = 100;
n_neuron_cluster = 10;
for i=1:length(n_classes)
    n_class = n_classes(i);
    % 加载数据集
    digits = 0:(n_class-1);
    data = load_mnist_data(digits,data_ratio);
    disp("load data success");
    for j=1:length(out_features_cpls)
        out_features_cpl = out_features_cpls(j);
        init_para = init_paramter(digits,in_features_cpl,out_features_cpl,n_neuron_cluster,verify_step,get_lr_step,get_lr_batch);
        disp("init para success")
        init_para
        rand('state',j);
        net = init_net(init_para);
        disp("init net success")
        [rl_check_points,rl_best_train_result] = rl_trainer(init_para,net,data,train_early_stopping,test_early_stopping);
        compare_class{i,j*2-1}=rl_check_points;
        compare_class{i,j*2}=rl_best_train_result.init_para;
    end
end
save(file_name,"compare_class","-v7.3");