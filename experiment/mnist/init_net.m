function net = init_net(init_para)
% cpl随机组合索引
net.rand_group_index_cpl = randperm(init_para.out_features_cpl);
%设置输入层和CPL之间的权重，会生成一个非均匀稀疏矩阵

net.weight_cpl = zeros(init_para.out_features_cpl, init_para.in_features_cpl);
num_prob = numel(init_para.sparse_prob);
for i = 1:init_para.n_neuron_cluster:init_para.out_features_cpl
    ind_cluster = net.rand_group_index_cpl(i:i+init_para.n_neuron_cluster-1);
    prob = init_para.sparse_prob(randi(num_prob));
    
    for k = 1:init_para.n_neuron_cluster
        net.weight_cpl(ind_cluster(k), :) = sprandn(1, init_para.in_features_cpl,prob);
    end
end
net.weight_cpl(net.weight_cpl~=0) = 1;

%设置CPL和输出层之间的权重
% shape(n_category,out_features_cpl)
net.weight_out = rand(init_para.n_category, init_para.out_features_cpl);
% shape(n_category,out_features_cpl)
net.weight_filter_out = zeros(size(net.weight_out));
net.weight_filter_out(net.weight_out>init_para.synaptic_th) = 1;