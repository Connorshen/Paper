function lr = get_lr(net,init_para,train_img,train_label)
batch_size = init_para.get_lr_batch;
[batch_img,batch_label] = get_batch(train_img,train_label,batch_size);
[b_output_cpl,~,~,~,~]=forward(net,init_para,batch_img',batch_label');
n_digit = length(init_para.digits);
output_cpl_map = cell(n_digit,1);
self_sum = cell(n_digit,1);
self_act_sum = cell(n_digit,1);
other_act_sum = cell(n_digit,1);
other_sum = cell(n_digit,1);
ratio = cell(n_digit,1);
lr = cell(n_digit,1);
for i=1:batch_size
    %这里加一是matlab的索引从1开始
    index = batch_label(i)+1;
    output_cpl = b_output_cpl(:,i)';
    output_cpl_map{index} = [output_cpl_map{index};output_cpl];
end
for i=1:n_digit
    neuron_sum = size(output_cpl_map{i},1);
    self_sum{i} = ones(1,init_para.out_features_cpl)*neuron_sum;
    other_sum{i} = ones(1,init_para.out_features_cpl)*(batch_size - neuron_sum);
end
for i=1:n_digit
    self_act_sum{i} = sum(output_cpl_map{i});
    other_act_sum{i} = sum(b_output_cpl') - self_act_sum{i};
end
for i=1:n_digit
    ratio{i} = self_act_sum{i}./self_sum{i}-other_act_sum{i}./other_sum{i};
end
for i=1:n_digit
    lr{i} = 1./(1+exp(-ratio{i}))*0.2;
end