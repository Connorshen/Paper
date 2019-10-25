function output_cpl = set_activity(input_cpl,init_para)
n_neuron_cluster = init_para.n_neuron_cluster;
out_features_cpl = init_para.out_features_cpl;
n_cluster = out_features_cpl/n_neuron_cluster;
batch_size = size(input_cpl,2);
input_cpl_group = reshape(input_cpl,n_neuron_cluster,n_cluster,batch_size);
[~,max_index] = max(input_cpl_group,[],1);
max_index = reshape(max_index,n_cluster,batch_size);
disp("1")