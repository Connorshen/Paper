function net = init_net(init_para,train_img,train_label)
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
% 随机设置权重初始值
% net.weight_out = rand(init_para.n_category, init_para.out_features_cpl);
% 根据激活比例来设置权重初始值
batch_size = 100;
net.weight_out = rand(init_para.n_category, init_para.out_features_cpl);
net.weight_filter_out = zeros(size(net.weight_out));
net.weight_filter_out(net.weight_out>init_para.synaptic_th) = 1;
[batch_img,batch_label] = get_batch(train_img,train_label,batch_size);
[~,b_output_cpl,~,~,~]=forward(net,init_para,batch_img',batch_label');
n_digit = length(init_para.digits);
output_cpl_map = cell(n_digit,1);
self_sum = sum(b_output_cpl,2)';
self_act_sum = cell(n_digit,1);
ratio = cell(n_digit,1);
for i=1:batch_size
    %这里加一是matlab的索引从1开始
    index = batch_label(i)+1;
    output_cpl = b_output_cpl(:,i)';
    output_cpl_map{index} = [output_cpl_map{index};output_cpl];
end
for i=1:n_digit
    self_act_sum{i} = sum(output_cpl_map{i});
end
for i=1:n_digit
    ratio{i} = self_act_sum{i}./self_sum;
end
net.weight_out = [];
for i=1:n_digit
    net.weight_out = [net.weight_out;ratio{i}];
end
net.weight_out(isnan(net.weight_out))=0;
% 大于阈值的设为1，小于阈值的置为0
net.weight_filter_out = zeros(size(net.weight_out));
net.weight_filter_out(net.weight_out>init_para.synaptic_th) = 1;