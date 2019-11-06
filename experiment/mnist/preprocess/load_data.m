function [train_img,train_label,test_img,test_label] = load_data()
test_img = load_image("t10k-images-idx3-ubyte");
test_label = load_label("t10k-labels-idx1-ubyte");
test_rand_index = randperm(size(test_img,1));
test_img=test_img(test_rand_index,:);
test_label=test_label(test_rand_index);
train_img = load_image("train-images-idx3-ubyte");
train_label = load_label("train-labels-idx1-ubyte");
train_rand_index = randperm(size(train_img,1));
train_img=train_img(train_rand_index,:);
train_label=train_label(train_rand_index);
end