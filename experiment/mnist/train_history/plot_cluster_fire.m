function plot_cluster_fire()
load("compare_cluster_fire.mat")
len = size(compare_cluster_fire,1);
rl_batch_train_result = compare_cluster_fire{len,4};
net = rl_batch_train_result.net;
init_para = rl_batch_train_result.init_para;
data_ratio = 0.1;% 数据集比例
early_stopping = 10;% 测试的时候提早break的step，不想提早结束的话取-1
data = load_mnist_data(init_para.digits,data_ratio);
[predict_result]=run_predicting(net,init_para,data,early_stopping);
input_cpl = predict_result.input_cpl;
img_origin = predict_result.img_origin;
labels = predict_result.label;
indexs = 1:size(labels,1);
digits = init_para.digits;
n_neuron_cluster = init_para.n_neuron_cluster;
rand_group_index_cpl = net.rand_group_index_cpl;
digit_cluster1 = [];
imgs_origin= [];
imgs_label = [];
for digit=digits
    ind = labels==digit;
    ind = indexs(ind);
    ind = ind(1);
    img = img_origin(ind,:);
    cpl = input_cpl(ind,:);
    label = labels(ind,:);
    cluster1_index = rand_group_index_cpl(1:n_neuron_cluster);
    cluster1 = cpl(cluster1_index);
    digit_cluster1 = [digit_cluster1;cluster1];
    imgs_origin = [imgs_origin;img];
    imgs_label = [imgs_label;label];
end
%imshow(reshape(imgs_origin(1,:),28,28));
figure(1);
set(gcf,'Position',[500,500,1200,200], 'color','w')
for digit=digits
    ind = digit+1;
    subplot(2,10,ind*2-1);
    imshow(reshape(imgs_origin(ind,:),28,28)')
    subplot(2,10,ind*2);
    bar(digit_cluster1(ind,:));
end