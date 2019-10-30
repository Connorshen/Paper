load("trained_weight_98_43.mat")
init_para = trained_weight.init_para;
net = trained_weight.net;
[~,~,test_img,test_label] = load_data(init_para.digits);
acc = run_testing(init_para,net,test_img,test_label,-1);