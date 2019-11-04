function [batch_img,batch_label] = get_batch(train_img,train_label,batch_size)
index = randperm(batch_size, batch_size);
batch_img = train_img(index,:);
batch_label = train_label(index,:);
