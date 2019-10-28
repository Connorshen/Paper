

function network = initial_network(init_para)

disp('initial network.')

% random group clusters in CPL
network.weight_recurrent_CPL = randperm(init_para.numNeurons_CPL);

% set weights between input and CLP
if init_para.flag_herg_inputCPL
    row = init_para.numNeurons_CPL;
    col = init_para.numNeurons_input;
    network.weight_input_CPL = zeros(row, col);
    num_prob = numel(init_para.hprob_input_CPL);
    for i = 1:init_para.numNeurons_cluster:init_para.numNeurons_CPL
        ind_cluster = network.weight_recurrent_CPL(i:i+init_para.numNeurons_cluster-1);
        prob = init_para.hprob_input_CPL(randi(num_prob));
        
        for k = 1:init_para.numNeurons_cluster
            network.weight_input_CPL(ind_cluster(k), :) = sprandn(1, col,prob);
        end
    end
else
    network.weight_input_CPL = full(sprandn(init_para.numNeurons_CPL, init_para.numNeurons_input,...
                       init_para.prob_input_CPL));
end
network.weight_input_CPL(find(network.weight_input_CPL~=0)) = 1;


% set the output weights
network.weight_CPL_decision = rand(init_para.numNeurons_decision, init_para.numNeurons_CPL);
% % nrom all columns
% for ic = 1:init_para.numNeurons_CPL
%     network.weight_CPL_decision(:, ic) = network.weight_CPL_decision(:, ic)./norm(network.weight_CPL_decision(:, ic));
% end

network.weightFilter_CPL_decision = zeros(size(network.weight_CPL_decision));
network.weightFilter_CPL_decision(find(network.weight_CPL_decision>0.8)) = 1;
