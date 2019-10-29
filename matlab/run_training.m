close all
clear

init_para = init_paramter();
net = init_net(init_para);
[train_img,train_label,test_img,test_label] = load_data(init_para.digits);
n_batch = idivide(int32(length(train_img)),int32(init_para.batch_size),"ceil");
batch_size = init_para.batch_size;
epoch = init_para.epoch;
train_result = zeros(n_batch*epoch,2);
step_all = 1;
for i=1:epoch
    for j=1:n_batch
        start_index = (j-1)*batch_size+1;
        end_index = start_index+batch_size-1;
        if end_index > length(train_img)
            end_index = length(train_img);
        end
        real_batch_size = end_index-start_index+1;
        % img,label
        batch_img = train_img(start_index:end_index,:)';
        batch_label = train_label(start_index:end_index,:)';
        %forward
        %shape(out_features_cpl,batch_size)
        input_cpl = net.weight_cpl*batch_img;
        %shape(out_features_cpl,batch_size)
        output_cpl = set_activity(input_cpl,init_para,net);
        %shape(n_category,batch_size)
        batch_out = net.weight_filter_out * output_cpl;
        %shape(n_category,batch_size)
        batch_predict_prob = exp(batch_out*init_para.gain_decision)./sum(exp(batch_out*init_para.gain_decision));
        [b_predict_prob, b_predict] = max(batch_predict_prob);  
        b_digit_category = b_predict - 1;
        %backward
        %shape(1,batch_size)
        b_reward = zeros(size(b_digit_category));
        b_reward(b_digit_category==batch_label)=1;
        for k=1:real_batch_size
            % update the weights on the final layer
            reward = b_reward(1,k);
            predict_prob = b_predict_prob(1,k);
            % shape(1,out_features_cpl)
            modify_weight = net.weight_out(b_predict(1,k), :);  % which synapses will be updated
            % shape(1,out_features_cpl)
            need_modify_weight = rand(1,init_para.out_features_cpl)<modify_weight;
            % shape(1,out_features_cpl)
            potential = output_cpl(:,k)'.* need_modify_weight;
            depress = ~output_cpl(:,k)'.* (rand(1,init_para.out_features_cpl)<0.01);
            if reward
                modify_weight = modify_weight + init_para.learning_rate*(reward - predict_prob)*(potential-depress);
            else
                modify_weight = modify_weight - init_para.learning_rate*predict_prob*potential;
            end
            modify_weight = max(modify_weight, 0);
            net.weight_out(b_predict(1,k), :) = modify_weight;
            net.weight_filter_out(b_predict(1,k), :) = net.weight_out(b_predict(1,k), :)>init_para.synaptic_th;
        end
        train_result(step_all,:) = [mean(b_reward),mean(b_predict_prob)];
        if rem(step_all,60)==0 && step_all>0
            % acc = run_testing(init_para,net,test_img,test_label,10);
            acc = mean(train_result(step_all-60+1:step_all,1));
            prob = mean(train_result(step_all-60+1:step_all,2));
            sprintf("epoch:%d | step:%d | acc:%.4f | prob:%.4f",i,step_all/60,acc,prob)
        end
        step_all = step_all+1;
    end
    save("history","history")
end