function output_CPL = set_activity_CPL(input_CPL, weight_recurrent_CPL, para)

numNeurons_CPL = para(1);
numNeurons_cluster = para(2);
output_CPL = zeros(size(input_CPL));
% 最大值置为1，其余为0
for i = 1:numNeurons_cluster:numNeurons_CPL
    ind_cluster = weight_recurrent_CPL(i:i+numNeurons_cluster-1);
    [~, ind_max] = max(input_CPL(ind_cluster));
    output_CPL(ind_cluster(ind_max)) = 1;
end