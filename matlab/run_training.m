clear all
load("test.mat")
load("train.mat")

init_para = init_paramter();
net = init_net(init_para);
n_batch = idivide(int32(length(train_img)),int32(init_para.batch_size),"ceil");
batch_size = init_para.batch_size;
epoch = init_para.epoch;
for i=1:epoch
    for j=1:n_batch
        start_index = (j-1)*batch_size+1;
        end_index = start_index+batch_size-1;
        if end_index > length(train_img)
            end_index = length(train_img);
        end
        batch_img = train_img(start_index:end_index,:);
        batch_label = train_label(start_index:end_index,:);
        %shape(out_features_cpl,batch_size)
        input_cpl = net.weight_cpl*batch_img';
        output_cpl = set_activity(input_cpl,init_para);
    end
end