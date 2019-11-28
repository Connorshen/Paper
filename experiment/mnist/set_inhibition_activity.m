function output_cpl = set_inhibition_activity(input_cpl,init_para,net)
n_neuron_cluster = init_para.n_neuron_cluster;
out_features_cpl = init_para.out_features_cpl;
n_cluster = out_features_cpl/n_neuron_cluster;
batch_size = size(input_cpl,2);

% 随机组合
base_index_rand = repmat(linspace(0,out_features_cpl*(batch_size-1),batch_size),out_features_cpl,1);
base_index_rand = reshape(base_index_rand,[],1);
rand_group_index_cpl = repmat(net.rand_group_index_cpl',batch_size,1);
rand_group_index_cpl = rand_group_index_cpl + base_index_rand;
input_cpl_rand = input_cpl(rand_group_index_cpl);
%shape(n_neuron_cluster,n_cluster,batch_size)
input_cpl_group = reshape(input_cpl_rand,n_neuron_cluster,n_cluster,batch_size);
[~,max_index_local] = max(input_cpl_group,[],1);
max_index_local = reshape(max_index_local,n_cluster*batch_size,1);
base_index = linspace(0,out_features_cpl*batch_size-n_neuron_cluster,n_cluster*batch_size)';
%shape(n_cluster*batch_size,1)
max_index = base_index + max_index_local;
%shape(out_features_cpl,batch_size)
output_cpl=zeros(size(input_cpl));
output_cpl(rand_group_index_cpl(max_index)) = 1;