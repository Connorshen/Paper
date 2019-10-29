
function [training_result, network_trained] = run_training( network_init,  init_para)

training_result = [];

% get all training data set
[train_img,train_label,~,~] = load_data(init_para.digit_label);
trials_round = init_para.trials_round;
step_all = 1;

for i = 1:init_para.num_rounds
    disp('the round training...')
    disp(i)
    num_trials = init_para.trials_round;
    % prepare training set for each round
    training_data = train_img((i*trials_round-trials_round)+1:i*trials_round, :);
    training_label = train_label((i*trials_round-trials_round)+1:i*trials_round, :);
    
    result_round = zeros(num_trials,4);
    
    for j = 1:num_trials
        label = training_label(j, 1);
        digit_img  = training_data(j,:)';
        
        input_CPL = network_init.weight_input_CPL * digit_img;
        output_CPL = set_activity_CPL(input_CPL, network_init.weight_recurrent_CPL, [init_para.numNeurons_CPL,init_para.numNeurons_cluster]);
                                    
        input_decision = network_init.weightFilter_CPL_decision * output_CPL;
        prob_list_decision = exp(input_decision*init_para.gain_decision)./sum(exp(input_decision*init_para.gain_decision));
        
        [prob_decision, ind_decision] = max(prob_list_decision);  
        digit_decision = ind_decision - 1;
        if digit_decision == label
            reward = 1;
        else
            reward = 0;
        end
        
        % update the weights on the final layer

        wm = network_init.weight_CPL_decision(ind_decision, :);  % which synapses will be updated
        num_wm = numel(wm);
        act_am = rand(1,num_wm)<wm;
        
        val_potential = output_CPL'.* act_am;
        val_depress = ~output_CPL'.* (rand(1,num_wm)<0.01);

        if reward
            wm = wm + 0.1*(reward - prob_decision).*(val_potential-val_depress);
        else
            wm = wm - 0.1*val_potential*prob_decision;
        end
        % all weights between 0 and 1
        wm = max(wm, 0);
        
        network_init.weight_CPL_decision(ind_decision, :) = wm;

        network_init.weightFilter_CPL_decision(ind_decision, :) = network_init.weight_CPL_decision(ind_decision, :)>init_para.synaptic_th; 
        
        % list results
        result_round(j, :) = [label, digit_decision, reward, prob_decision];
        step_all = step_all +1;
    end
    disp('training result in a round...');
    disp(mean(result_round(:, 3:4)));
    
    training_result = [training_result; result_round];   
end

network_trained = network_init;


