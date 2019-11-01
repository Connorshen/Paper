function [acc,loss] = run_testing(net,init_para,data,early_stopping)
test_img = data.test_img;
test_label = data.test_label;
batch_size = 100;
test_len = int32(size(test_img,1));
n_batch = idivide(test_len,batch_size,"ceil");
reward_all = [];
log_all = [];
for i=1:n_batch
    start_index = (i-1)*batch_size+1;
    end_index = start_index+batch_size-1;
    if end_index > test_len
        end_index = test_len;
    end
    % img,label
    batch_img = test_img(start_index:end_index,:)';
    batch_label = test_label(start_index:end_index,:)';
    %forward
    [~,b_reward,~,b_predict_prob,~]=forward(net,init_para,batch_img,batch_label);
    reward_all = [reward_all;b_reward'];
    log_all = [log_all;sum(-log(b_predict_prob))];
    if i==early_stopping
        break
    end
end
acc = mean(reward_all);
loss = mean(log_all);