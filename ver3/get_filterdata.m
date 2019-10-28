
function [data_ind, data] = get_filterdata(digit_label, type)

data_ind = [];
data = [];
i_label = 1;

for label = digit_label
    filename = ['../filterdata/' type int2str(label)];
    load(filename);
    num_digits = size(D_filtered, 1);
    
    for n = 1:num_digits
        ind_correct = 0;
        data_ind(end+1, :) = [label, i_label, ind_correct];
        i_label = i_label + 1;
    end
    data = [data;D_filtered];
    clear D_filtered;
end

num_digit_data = size(data_ind, 1);
data_ind = data_ind(randperm(num_digit_data)', :);