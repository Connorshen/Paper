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
output_cpl = predict_result.output_cpl;
img_origin = predict_result.img_origin;
labels = predict_result.label;
indexs = 1:size(labels,1);
digits = init_para.digits;
n_neuron_cluster = init_para.n_neuron_cluster;
rand_group_index_cpl = net.rand_group_index_cpl;
digits_cluster1 = [];
digit_cluster1_fire_ratio = [];
imgs_origin= [];
imgs_label = [];
cluster1_fire_all = [];
for digit=digits
    ind = labels==digit;
    ind = indexs(ind);
    output_cpl_digit = output_cpl(ind,:);
    output_cpl_digit = sum(output_cpl_digit);
    ind = ind(1);
    img = img_origin(ind,:);
    i_cpl = input_cpl(ind,:);
    o_cpl = output_cpl(ind,:);
    label = labels(ind,:);
    cluster1_index = rand_group_index_cpl(1:n_neuron_cluster);
    cluster1 = i_cpl(cluster1_index);
    cluster1_fire_digit_all = output_cpl_digit(cluster1_index);
    cluster1_fire_all = [cluster1_fire_all;cluster1_fire_digit_all];
    cluster1_fire_ratio = cluster1_fire_digit_all/sum(cluster1_fire_digit_all);
    digits_cluster1 = [digits_cluster1;cluster1];
    digit_cluster1_fire_ratio=[digit_cluster1_fire_ratio;cluster1_fire_ratio];
    imgs_origin = [imgs_origin;img];
    imgs_label = [imgs_label;label];
end
cluster1_fire_all = sum(cluster1_fire_all);
cluster1_fire_all_ratio = cluster1_fire_all/sum(cluster1_fire_all);

figure(1);
set(gcf,'Position',[500,500,1200,200], 'color','w')
for digit=digits
    ind = digit+1;
    subplot(2,10,ind*2-1);
    imshow(reshape(imgs_origin(ind,:),28,28))
    subplot(2,10,ind*2);
    digit_cluster1 = digits_cluster1(ind,:);
    [~, max_ind] = max(digit_cluster1);
    digit_cluster1 = diag(digit_cluster1);
    bar_fig = bar(digit_cluster1,"stack");
    for i = 1:n_neuron_cluster
        set(bar_fig(i),"FaceColor","b");
    end
    set(bar_fig(max_ind),"FaceColor","r");
    axis([-inf,inf,1,4])
    xlabel("pos")
    ylabel("value")
end
figure(2);
set(gcf,'Position',[500,300,1200,200], 'color','w')
for digit=digits
    ind = digit+1;
    subplot(2,10,ind*2-1);
    imshow(reshape(imgs_origin(ind,:),28,28))
    subplot(2,10,ind*2);
    bar(digit_cluster1_fire_ratio(ind,:));
    axis([-inf,inf,0,0.6])
    xlabel("pos")
    ylabel("fire ratio")
end
figure(3);
set(gcf,'Position',[500,300,400,300], 'color','w')
bar(cluster1_fire_all_ratio);
xlabel("pos");
ylabel("fire prob");
title("cluster1 fire ratio");