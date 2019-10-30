function run_training(init_para,net)

[train_img,train_label,test_img,test_label] = load_data(init_para.digits,0.1);
disp("load data success");
train_len = size(train_img,1);
epoch = init_para.epoch;
n_digit = length(init_para.digits);
% step reward prob verify_acc
check_points = zeros(train_len*epoch,4);
verify_step = init_para.verify_step;
get_lr_step = init_para.get_lr_step;
fprintf("train_len:%d \n",train_len)
for i=1:epoch
    learning_rate = init_para.learning_rate;
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
        lr = learning_rate{batch_label+1};
        % shape(1,out_features_cpl)
        modify_weight = net.weight_out(predict, :);  % which synapses will be updated
        % shape(1,out_features_cpl)
        need_modify_weight = rand(1,init_para.out_features_cpl)<modify_weight;
        % shape(1,out_features_cpl)
        potential = output_cpl'.* need_modify_weight;
        depress = ~output_cpl'.* (rand(1,init_para.out_features_cpl)<0.01);
        if reward
            modify_weight = modify_weight + (reward - predict_prob)*lr.*(potential-depress);
        else
            modify_weight = modify_weight - predict_prob*lr.*potential;
        end
        modify_weight = max(modify_weight, 0);
        net.weight_out(predict, :) = modify_weight;
        net.weight_filter_out(predict, :) = net.weight_out(predict, :)>init_para.synaptic_th;
        check_points(j,:) = [j,reward,predict_prob,0];
        % 在测试集上验证
        if rem(j,verify_step)==0
            start_index = j-verify_step+1;
            end_index = j;
            acc = run_testing(net,init_para,test_img,test_label,10);
            % acc = mean(check_points(start_index:end_index,2));
            prob = mean(check_points(start_index:end_index,3));
            check_points(j,4)=acc;
            fprintf("epoch:%d | step:%d | acc:%.4f | prob:%.4f\n",i,j,acc,prob)
        end
        % 更新学习率
        if rem(j,get_lr_step)==0
            learning_rate = get_lr(net,init_para,train_img,train_label);
        end
    end
    save("check_points","check_points")
end