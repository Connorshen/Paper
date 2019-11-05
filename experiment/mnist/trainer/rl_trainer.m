function [check_points,best_train_result] = rl_trainer(init_para,net,data,train_early_stopping,test_early_stopping)

train_img = data.train_img;
train_label = data.train_label;
train_len = size(train_img,1);
epoch = init_para.epoch;
% step reward prob verify_acc
check_points = zeros(train_len*epoch,5);
verify_step = init_para.verify_step;
fprintf("train_len:%d \n",train_len)
best_net = net;
best_acc = -inf;
for i=1:epoch
    for j=1:train_len
        % img,label
        batch_img = train_img(j,:)';
        batch_label = train_label(j,:)';
        %forward
        %shape(out_features_cpl,batch_size)
        [input_cpl,output_cpl,b_reward,~,b_predict_prob,b_predict,~]=forward(net,init_para,batch_img,batch_label);
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
        check_points(j,:) = [j,reward,predict_prob,0,0];
        if rem(j,verify_step)==0 || j==1
            start_index = max(j-verify_step+1,1);
            end_index = j;
            [acc,loss] = run_testing(net,init_para,data,test_early_stopping);
            prob = mean(check_points(start_index:end_index,3));
            check_points(j,4)=acc;
            check_points(j,5)=loss;
            fprintf("epoch:%d | step:%d | acc:%.4f | prob:%.4f\n",i,j,acc,prob)
            if best_acc<acc
                best_acc = acc;
                best_net = net;
            end
        end
        if j==train_early_stopping
            break
        end
    end
end
init_para.trainer = "rl_trainer";
best_train_result.net = best_net;
best_train_result.init_para = init_para;