function set_image_feature()
% transform image digits to features according to garbar filters

% path where MinMaxFilterFolder is found. This function implements local maxima selection,
path(path,'MinMaxFilterFolder/MinMaxFilterFunctions/');

[train_img,train_label,test_img,test_label] = load_data();
train_img=train_img(1:5,:);
train_label=train_label(1:5,:);
test_img=test_img(1:5,:);
test_label=test_label(1:5,:);


train_img_filter = get_filtered_feature(train_img);
test_img_filter = get_filtered_feature(test_img);
train_label=train_label;
train_img_origin = train_img;
train_img=train_img_filter;
test_label=test_label;
test_img_origin = test_img;
test_img=test_img_filter;

save("train","train_img","train_label","train_img_origin");
save("test","test_img","test_label","test_img_origin");