clc
clear
load("results_trained_0_9_p98-35.mat")
init_para = init_paramter();
net = init_net(init_para);
[~,~,test_img,test_label] = load_data(init_para.digits);
net.rand_group_index_cpl = results.network_trained.weight_recurrent_CPL;
net.weight_cpl = results.network_trained.weight_input_CPL;
net.weight_out = results.network_trained.weight_CPL_decision;
net.weight_filter_out = results.network_trained.weightFilter_CPL_decision;
run_testing(init_para,net,test_img,test_label,10);