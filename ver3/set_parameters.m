

function  init_para = set_parameters()

init_para.num_rounds = 1000;
init_para.trials_round = 60;
init_para.flag_herg_inputCPL = true;
init_para.flag_saveresult = true;

init_para.digit_label = 0:9; 

init_para.numNeurons_input = 2560;
init_para.numNeurons_CPL = 200000;
init_para.numNeurons_cluster = 10;
init_para.numNeurons_decision = 10;

init_para.prob_input_CPL = 0.01;
init_para.hprob_input_CPL = 0.001:0.001:0.01;

init_para.diff_th = 0.1;
init_para.learning_rate = 0.5;
init_para.gain_decision = 0.01;
init_para.synaptic_th = 0.8;
init_para.potential_prob = 0.95;




