
function [syn_A_begin_list, syn_B_begin_list, syn_A_end_list, syn_B_end_list, p_update_list] = matching()
         
N         = 800;            % number of previous neurons;
nwin      = 20;             % width of history window
nSubject  = 100;
nTrials   = 500;
maxReward = 200;
minReward = 20;

eta       = 0.01;          % learning rate
xigma     = 1.68;           % for probability of choice

% list of outcomes
pA_list = zeros(nSubject, nTrials);
reward_listA = zeros(nSubject, nTrials);
reward_listB = zeros(nSubject, nTrials);
action_list = zeros(nSubject, nTrials); 
actionA = zeros(nSubject, nTrials);
actionB = zeros(nSubject, nTrials);
p_update_list = zeros(nSubject, nTrials);
syn_A_begin_list = zeros(nSubject, N);
syn_B_begin_list = zeros(nSubject, N);
syn_A_end_list = zeros(nSubject, N);
syn_B_end_list = zeros(nSubject, N);

for j = 1:nSubject

    iniAction = rand(nwin, 1)>0.5;      % for initial reward schedule
    
    pre_reward = minReward + (maxReward-minReward)*rand;           % expected reward (for simplicity, we use previous reward obtained by previous action
    
    syn_A = rand(N, 1);                 % synapses between population A and previous neurons, 1 means potential, 0 means depression
    syn_B = rand(N, 1);                 % synapses between population B and previous neurons
    
    syn_A = syn_A./norm(syn_A);         % norminize the weights
    syn_B = syn_B./norm(syn_B);         % norminize the weights
    
    syn_A_begin_list(j, :) = syn_A';
    syn_B_begin_list(j, :) = syn_B';
    for i = 1:nTrials
        % compute the policy according weights
        deltaI = sum(syn_A) - sum(syn_B);
        pA = 1/(1+exp(-deltaI/xigma));% probability choosing target A
        % take action according policy
        action = pA > rand;              % action is 1 if target A is selected
        
        pA_list(j, i) = pA;
        action_list(j, i) = action;  
        
        % get reward according the current action and reward schedule
        r = getReward(action_list, i, j, iniAction);  
        deltaR = r - pre_reward;
        % learning algorithem
        p_update = tanh(deltaR*0.01);
        if(p_update<0)
            p_update = p_update*-1;
        elseif(p_update==0)
            p_update = 0.1;
        end
        p_update_list(j, i) = p_update;
        plusone = eta*(rand(N, 1)<p_update)*diffr(deltaR);
        
        if action > 0                   % the target A is chosen
            syn_A = syn_A + plusone;
            % remove exceptional values of synapses
            syn_A(find(syn_A<0)) = 0;
            syn_A(find(syn_A>1)) = 1;
            reward_listA(j, i) = r;
            actionA(j, i) = 1;
            syn_A = syn_A./norm(syn_A);

        else
            syn_B = syn_B + plusone; 
            % remove exceptional values of synapses
            syn_B(find(syn_B<0)) = 0;
            syn_B(find(syn_B>1)) = 1;
            reward_listB(j, i) = r;
            actionB(j, i) = 1;
            syn_B = syn_B./norm(syn_B);
        end   
        
        pre_reward = r;

    end
    
    syn_A_end_list(j, :) = syn_A';
    syn_B_end_list(j, :) = syn_B';
    
end
