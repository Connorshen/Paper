function init_para = init_paramter(digits,in_features_cpl,out_features_cpl,n_neuron_cluster,verify_step,get_lr_step,get_lr_batch)
init_para.epoch=1;
init_para.verify_step = verify_step;
%每隔get_lr_step次，获取一次lr
init_para.get_lr_step = get_lr_step;
%每次获取lr的batch的大小为get_lr_batch
init_para.get_lr_batch = get_lr_batch;
init_para.digits = digits;

init_para.in_features_cpl = in_features_cpl;
init_para.out_features_cpl = out_features_cpl;
init_para.n_neuron_cluster = n_neuron_cluster;
init_para.n_category = length(init_para.digits);
init_para.sparse_prob = 0.001:0.001:0.01;

init_para.weight_density_cpl = 0.01;
init_para.learning_rate = 0.1;
init_para.gain_decision = 0.01;
init_para.synaptic_th = 0.8;

learning_rate = cell(init_para.n_category,1);
for i=1:init_para.n_category
    learning_rate{i} = ones(1,init_para.out_features_cpl)*init_para.learning_rate;
end
init_para.learning_rate_batch = learning_rate;