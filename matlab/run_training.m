function run_training(init_para,net)

[train_img,train_label,test_img,test_label] = load_data(init_para.digits,0.1);
train_len = size(train_img,1);
epoch = init_para.epoch;
% step reward prob verify_acc
check_points = zeros(train_len*epoch,4);
verify_step = init_para.verify_step;
for i=1:epoch
    for j=1:train_len
        % img,label
        batch_img = train_img(j,:)';
        batch_label = train_label(j,:)';
        %forward
        %shape(out_features_cpl,batch_size)
        [output_cpl,b_reward,~,b_predict_prob,b_predict]=forward(net,init_para,batch_img,batch_label);
        %backward
        % update the weights on the final layer
        reward = b_reward;
        predict_prob = b_predict_prob;
        predict = b_predict;
        % shape(1,out_features_cpl)
        modify_weight = net.weight_out(predict, :);  % which synapses will be updated
        % shape(1,out_features_cpl)
        need_modify_weight = rand(1,init_para.out_features_cpl)<modify_weight;
        % shape(1,out_features_cpl)
        potential = output_cpl'.* need_modify_weight;
        depress = ~output_cpl'.* (rand(1,init_para.out_features_cpl)<0.01);
        if reward
            modify_weight = modify_weight + init_para.learning_rate*(reward - predict_prob)*(potential-depress);
        else
            modify_weight = modify_weight - init_para.learning_rate*predict_prob*potential;
        end
        modify_weight = max(modify_weight, 0);
        net.weight_out(predict, :) = modify_weight;
        net.weight_filter_out(predict, :) = net.weight_out(predict, :)>init_para.synaptic_th;
        check_points(j,:) = [j,reward,predict_prob,0];
        if rem(j,verify_step)==0
            start_index = j-verify_step+1;
            end_index = j;
            acc = run_testing(init_para,net,test_img,test_label,-1);
            % acc = mean(check_points(start_index:end_index,2));
            prob = mean(check_points(start_index:end_index,3));
            sprintf("epoch:%d | step:%d | acc:%.4f | prob:%.4f",i,j,acc,prob)
        end
    end
    save("check_points","check_points")
end