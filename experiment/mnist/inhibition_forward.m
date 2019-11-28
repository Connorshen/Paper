function [input_cpl,output_cpl,b_reward,b_digit_category,b_predict_prob,b_predict,b_predict_prob_all]=inhibition_forward(net,init_para,batch_img,batch_label)
%shape(out_features_cpl,batch_size)
input_cpl = net.weight_cpl*batch_img;
%shape(out_features_cpl,batch_size)
output_cpl = set_inhibition_activity(input_cpl,init_para,net);
%shape(n_category,batch_size)
batch_out = net.weight_filter_out * output_cpl;
%shape(n_category,batch_size)
batch_predict_prob = exp(batch_out*init_para.gain_decision)./sum(exp(batch_out*init_para.gain_decision));
[b_predict_prob, b_predict] = max(batch_predict_prob);
b_digit_category = b_predict - 1;
b_reward = zeros(size(b_digit_category));
b_reward(b_digit_category==batch_label)=1;
b_predict_prob_all = batch_predict_prob;