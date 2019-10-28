function testing_result = run_testing( network_trained, init_para)

% get all training data set
[ind_digit_data, digit_data] = get_filterdata(init_para.digit_label, 'test');
num_digit_data = size(ind_digit_data, 1);

testing_result = zeros(num_digit_data, 5);

disp('start testing...')
for j = 1:num_digit_data
    
    label = ind_digit_data(j, 1);
    ind_label = ind_digit_data(j, 2);
    digit_img  = digit_data(ind_label,:)';

    input_CPL = network_trained.weight_input_CPL * digit_img;
    output_CPL = set_activity_CPL(input_CPL, network_trained.weight_recurrent_CPL, [init_para.numNeurons_CPL,init_para.numNeurons_cluster, init_para.flag_sparse, init_para.diff_th]);

    input_decision = network_trained.weightFilter_CPL_decision * output_CPL;
    prob_list_decision = exp(input_decision*init_para.gain_decision)./sum(exp(input_decision*init_para.gain_decision));
     
    [prob_decision, ind_decision] = max(prob_list_decision);  
    digit_decision = ind_decision - 1;
    if digit_decision == label
        reward = 1;
    else
        reward = 0;
    end
    
    testing_result(j, :) = [ind_label, label, digit_decision, reward, prob_decision];
    
    if reward == 0
        show_error_image(init_para, testing_result(j, :))
    end
end

disp('testing result ...');
disp(mean(testing_result(:, 3:4)));
    
    