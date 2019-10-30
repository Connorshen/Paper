function init_para = init_paramter()
init_para.epoch=1;
init_para.verify_step = 100;
init_para.digits = 0:9; 

init_para.in_features_cpl = 2560;
init_para.out_features_cpl = 200000;
init_para.n_neuron_cluster = 10;
init_para.n_category = length(init_para.digits);
init_para.sparse_prob = 0.001:0.001:0.01;

init_para.weight_density_cpl = 0.01;
init_para.learning_rate = 0.1;
init_para.gain_decision = 0.01;
init_para.synaptic_th = 0.8;
