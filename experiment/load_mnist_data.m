function data = load_mnist_data(digits,ratio)
load("test_origin.mat")
load("train_origin.mat")
train_img = double(train_img);
test_img = double(test_img);
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
if ratio>0
    train_len = size(train_img,1);
    test_len = size(test_img,1);
    train_img = train_img(1:int32(train_len*ratio),:);
    train_label = train_label(1:int32(train_len*ratio),:);
    test_img = test_img(1:int32(test_len*ratio),:);
    test_label = test_label(1:int32(test_len*ratio),:);
end
data.train_img=train_img;
data.train_label=train_label;
data.test_img=test_img;
data.test_label=test_label;
