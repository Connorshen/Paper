function acc = run_testing(net,test_img,test_label,early_stopping)
batch_size = 20;
n_batch = idivide(int32(length(test_img)),batch_size,"ceil");
reward_all = [];
for i=1:n_batch
    start_index = (i-1)*batch_size+1;
    end_index = start_index+batch_size-1;
    if end_index > length(test_img)
        end_index = length(test_img);
    end
    % img,label
    batch_img = test_img(start_index:end_index,:)';
    batch_label = test_label(start_index:end_index,:)';
    %forward
    [~,b_reward,~,~]=forward(net,batch_img,batch_label);
    reward_all = [reward_all;b_reward'];
    if i==early_stopping
        break
    end
end
acc = mean(reward_all);