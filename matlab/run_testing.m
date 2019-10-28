function acc = run_testing(init_para,net,test_img,test_label,early_stopping)
n_batch = idivide(int32(length(test_img)),int32(init_para.batch_size),"ceil");
batch_size = init_para.batch_size;
reward_all = [];
for j=1:n_batch
    start_index = (j-1)*batch_size+1;
    end_index = start_index+batch_size-1;
    if end_index > length(test_img)
        end_index = length(test_img);
    end
    % img,label
    batch_img = test_img(start_index:end_index,:)';
    batch_label = test_label(start_index:end_index,:)';
    %forward
    %shape(out_features_cpl,batch_size)
    input_cpl = net.weight_cpl*batch_img;
    %shape(out_features_cpl,batch_size)
    output_cpl = set_activity(input_cpl,init_para);
    %shape(n_category,batch_size)
    batch_out = net.weight_filter_out * output_cpl;
    %shape(n_category,batch_size)
    batch_predict_prob = exp(batch_out*init_para.gain_decision)./sum(exp(batch_out*init_para.gain_decision));
    [~, b_predict] = max(batch_predict_prob);  
    b_digit_category = b_predict - 1;
    b_reward = zeros(size(b_digit_category));
    b_reward(b_digit_category==batch_label)=1;
    reward_all = [reward_all;b_reward'];
    if j==early_stopping
        break
    end
end
acc = mean(reward_all);