function plot_predict()
load("compare_convergence.mat")
len = size(compare_convergence,1);
rl_batch_train_result = compare_convergence{len,4};
net = rl_batch_train_result.net;
init_para = rl_batch_train_result.init_para;
data_ratio = 0.1;% 数据集比例
early_stopping = 10;% 测试的时候提早break的step，不想提早结束的话取-1
data = load_mnist_data(init_para.digits,data_ratio);
[predict_result]=run_predicting(net,init_para,data,early_stopping);
disp(1)