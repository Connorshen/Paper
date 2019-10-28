function output_CPL = set_activity_CPL(input_CPL, weight_recurrent_CPL, para)

numNeurons_CPL = para(1);
numNeurons_cluster = para(2);
output_CPL = zeros(size(input_CPL));
% input_CPL(weight_recurrent_CPL') = input_CPL;
%n_cluster = numNeurons_CPL/numNeurons_cluster;
% 最大值置为1，其余为0
%input_cpl_group = reshape(input_CPL,numNeurons_cluster,n_cluster);
%[~,max_index_local] = max(input_cpl_group,[],1);
%max_index_local = reshape(max_index_local,n_cluster,1);
%base_index = linspace(0,numNeurons_CPL-numNeurons_cluster,n_cluster)';
%max_index = base_index + max_index_local;
%output_CPL(max_index) = 1;
% 最大值置为1，其余为0
for i = 1:numNeurons_cluster:numNeurons_CPL
    ind_cluster = weight_recurrent_CPL(i:i+numNeurons_cluster-1);
    [~, ind_max] = max(input_CPL(ind_cluster));
    output_CPL(ind_cluster(ind_max)) = 1;
end