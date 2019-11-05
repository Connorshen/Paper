function [predict_result]=run_predicting(net,init_para,data,early_stopping)
test_img = data.test_img;
test_label = data.test_label;
test_img_origin = data.test_origin;
batch_size = 100;
test_len = int32(size(test_img,1));
n_batch = idivide(test_len,batch_size,"ceil");
predict_all = [];
reward_all = [];
predict_prob_all = [];
input_cpl_all = [];
label_all = [];
test_img_origin_all = [];
for i=1:n_batch
    start_index = (i-1)*batch_size+1;
    end_index = start_index+batch_size-1;
    if end_index > test_len
        end_index = test_len;
    end
    % img,label
    batch_img = test_img(start_index:end_index,:)';
    batch_label = test_label(start_index:end_index,:)';
    batch_img_origin = test_img_origin(start_index:end_index,:);
    %forward
    [input_cpl,~,b_reward,b_digit_category,~,~,b_predict_prob_all]=forward(net,init_para,batch_img,batch_label);
    predict_all = [predict_all;b_digit_category'];
    reward_all = [reward_all;b_reward'];
    predict_prob_all = [predict_prob_all;b_predict_prob_all'];
    input_cpl_all = [input_cpl_all;input_cpl'];
    label_all = [label_all;batch_label'];
    test_img_origin_all = [test_img_origin_all;batch_img_origin];
    if i==early_stopping
        break
    end
end
predict_result.predict = predict_all;
predict_result.reward = reward_all;
predict_result.predict_prob = predict_prob_all;
predict_result.input_cpl = input_cpl_all;
predict_result.label = label_all;
predict_result.img_origin = test_img_origin_all;