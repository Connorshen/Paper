function set_image_feature()
% transform image digits to features according to garbar filters

% path where MinMaxFilterFolder is found. This function implements local maxima selection,
path(path,'MinMaxFilterFolder/MinMaxFilterFunctions/');

[train_img,train_label,test_img,test_label] = load_data();
num = 1000;
train_img=train_img(1:num,:);
train_label=train_label(1:num,:);
test_img=test_img(1:num,:);
test_label=test_label(1:num,:);


train_img_filter = get_filtered_feature(train_img);
test_img_filter = get_filtered_feature(test_img);
train_label=train_label;
train_img_origin = train_img;
train_img=train_img_filter;
test_label=test_label;
test_img_origin = test_img;
test_img=test_img_filter;

save("../train","train_img","train_label","train_img_origin");
save("../test","test_img","test_label","test_img_origin");