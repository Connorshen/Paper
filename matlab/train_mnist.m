clear
clc
init_para = init_paramter();
net = init_net(init_para);
run_training(init_para,net)