
function [pA_list,reward_listA, reward_listB] = matching_maxmizing(p_mean)

         
N         = 800;            % number of previous neurons;
nwin      = 20;             % width of history window
nSubject  = 200;
nTrials   = 1000;
maxReward = 200;
minReward = 20;

eta       = 0.01;          % learning rate
xigma     = 1.68;           % for probability of choice

if nargin < 1
    p_attention = 0.5;
end

% list of outcomes
pA_list = zeros(nSubject, nTrials);
reward_listA = zeros(nSubject, nTrials);
reward_listB = zeros(nSubject, nTrials);
action_list = zeros(nSubject, nTrials); 
actionA = zeros(nSubject, nTrials);
actionB = zeros(nSubject, nTrials);
p_update_list = zeros(nSubject, nTrials);

for j = 1:nSubject

    iniAction = rand(nwin, 1)>0.5;      % for initial reward schedule
    
    pre_reward = minReward + (maxReward-minReward)*rand;           % expected reward (for simplicity, we use previous reward obtained by previous action
    preA_reward = pre_reward;
    preB_reward = pre_reward;
    
    syn_A = rand(N, 1);                 % synapses between population A and previous neurons, 1 means potential, 0 means depression
    syn_B = rand(N, 1);                 % synapses between population B and previous neurons
    
    syn_A = syn_A./norm(syn_A);         % norminize the weights
    syn_B = syn_B./norm(syn_B);         % norminize the weights
    
    p_attention = p_mean + 0.2.*randn;
    if length(p_mean)>1
        if rand > 0.5
            p_attention = p_mean(1) + 0.2.*randn;
        else
            
            p_attention = p_mean(2) + 0.2.*randn;
        end
    end
    
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
        %xx = diff(deltaR);
        plusone = eta*(rand(N, 1)<p_update)*diffr(deltaR);
        
        if action > 0                   % the target A is chosen
            deltaRA = r - preA_reward;
            pA_update = tanh(deltaRA*0.1);
            if(pA_update<0)
                pA_update = pA_update*-1;
            elseif(pA_update==0)
                pA_update = 0.1;
            end
            plusoneA = eta*(rand(N, 1)<pA_update)*diffr(deltaRA);
            
            if(rand >p_attention)
                syn_A = syn_A + plusone;
            else
                syn_A = syn_A + plusoneA;
            end
            % remove exceptional values of synapses
            syn_A(find(syn_A<0)) = 0;
            syn_A(find(syn_A>1)) = 1;
            reward_listA(j, i) = r;
            actionA(j, i) = 1;
            syn_A = syn_A./norm(syn_A);
            preA_reward = r;

        else
            deltaRB = r - preB_reward;
            pB_update = tanh(deltaRB*0.1);
            if(pB_update<0)
                pB_update = pB_update*-1;
            elseif(pB_update==0)
                pB_update = 0.1;
            end
            plusoneB = eta*(rand(N, 1)<pB_update)*diffr(deltaRB);
            
            if(rand>p_attention)
                syn_B = syn_B + plusone; 
            else
                syn_B = syn_B + plusoneB; 
            end
            % remove exceptional values of synapses
            syn_B(find(syn_B<0)) = 0;
            syn_B(find(syn_B>1)) = 1;
            reward_listB(j, i) = r;
            actionB(j, i) = 1;
            syn_B = syn_B./norm(syn_B);
            preB_reward = r;
        end   
        
        pre_reward = r;

    end
    
end



