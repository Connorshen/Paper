function [train_img,train_label,test_img,test_label] = load_data(digits)
%load("test.mat")
%load("train.mat")
load("test_origin.mat")
load("train_origin.mat")
for i = 0:9
    if ~ismember(i,digits)
        index = train_label==i;
        train_img(index,:)=[];
        train_label(index,:)=[];
        index = test_label==i;
        test_img(index,:)=[];
        test_label(index,:)=[];
    end
end

