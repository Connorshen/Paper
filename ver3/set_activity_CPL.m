function output_CPL = set_activity_CPL(input_CPL, weight_recurrent_CPL, para)

numNeurons_CPL = para(1);
numNeurons_cluster = para(2);
flag_sparse = para(3);
diff_th = para(4);

output_CPL = zeros(size(input_CPL));
% 最大值置为1，其余为0
for i = 1:numNeurons_cluster:numNeurons_CPL
    
    ind_cluster = weight_recurrent_CPL(i:i+numNeurons_cluster-1);
    [val_sorted, ind_sorted] = sort(input_CPL(ind_cluster), 'descend');
    
    if flag_sparse
        % sparse according to the diff between the max and the second val
        if val_sorted(1)-val_sorted(2) > diff_th
            output_CPL(ind_cluster(ind_sorted(1))) = 1;
        end
    else
        output_CPL(ind_cluster(ind_sorted(1))) = 1;
    end
end