function net = init_net(init_para)
%设置输入层和CPL之间的权重
% shape(out_features_cpl,in_features_cpl)
net.weight_cpl = full(sprandn(init_para.out_features_cpl, init_para.in_features_cpl,init_para.weight_density_cpl));
net.weight_cpl(find(net.weight_cpl~=0)) = 1;
%设置CPL和输出层之间的权重
% shape(n_category,out_features_cpl)
net.weight_out = rand(init_para.n_category, init_para.out_features_cpl);
% shape(n_category,out_features_cpl)
net.weight_filter_out = zeros(size(net.weight_out));
net.weight_filter_out(find(net.weight_out>init_para.synaptic_th)) = 1;