function [b_output_cpl,b_reward,b_digit_category,b_predict_prob,b_predict]=forward(net,init_para,batch_img,batch_label)
%shape(out_features_cpl,batch_size)
input_cpl = net.weight_cpl*batch_img;
%shape(out_features_cpl,batch_size)
b_output_cpl = set_activity(input_cpl,init_para,net);
%shape(n_category,batch_size)
batch_out = net.weight_filter_out * b_output_cpl;
%shape(n_category,batch_size)
batch_predict_prob = exp(batch_out*init_para.gain_decision)./sum(exp(batch_out*init_para.gain_decision));
[b_predict_prob, b_predict] = max(batch_predict_prob);
b_digit_category = b_predict - 1;
b_reward = zeros(size(b_digit_category));
b_reward(b_digit_category==batch_label)=1;