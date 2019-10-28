function [data_ind, data] = get_testdata(digit_label)

data_ind = [];
data = [];
i_label = 1;

for label = digit_label
    filename = ['../testdata/' 'test' int2str(label)];
    load(filename);
    num_digits = size(D, 1);
    
    for n = 1:num_digits
        ind_correct = 0;
        data_ind(end+1, :) = [label, i_label, ind_correct];
        i_label = i_label + 1;
    end
    data = [data;D];
    clear D_filtered;
end