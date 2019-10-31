clear
clc
init_para = init_paramter();
disp("init para success")
init_para
net = init_net(init_para);
disp("init net success")
run_training(init_para,net);
plot_acc()