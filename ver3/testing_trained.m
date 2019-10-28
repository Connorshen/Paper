
clear all
load('results_trained_0_9_p98-35.mat', 'results')

network_trained = results.network_trained;
init_para = results.init_para;

testing_result = run_testing( network_trained, init_para);