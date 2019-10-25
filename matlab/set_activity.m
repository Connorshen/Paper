function output_cpl = set_activity(input_cpl,init_para)
n_neuron_cluster = init_para.n_neuron_cluster;
out_features_cpl = init_para.out_features_cpl;
n_cluster = out_features_cpl/n_neuron_cluster;
batch_size = size(input_cpl,2);
%shape(n_neuron_cluster,n_cluster,batch_size)
input_cpl_group = reshape(input_cpl,n_neuron_cluster,n_cluster,batch_size);
[~,max_index_local] = max(input_cpl_group,[],1);
max_index_local = reshape(max_index_local,n_cluster*batch_size,1);
base_index = linspace(0,out_features_cpl*batch_size-n_neuron_cluster,n_cluster*batch_size)';
%shape(n_cluster*batch_size,1)
max_index = base_index + max_index_local;
%shape(out_features_cpl,batch_size)
output_cpl=zeros(size(input_cpl));
output_cpl(max_index) = 1;