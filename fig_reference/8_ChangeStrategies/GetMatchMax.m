
function [choice_matching, choice_optimizing] = GetMatchMax (fileName)
% matching: choose the target can get bigger reward than other.

num = xlsread(fileName);    % read experimental data from a file
numTrials = 100;
win = 20;
numSubject = size(num, 1);          % the number of subjects
choice_matching = zeros(numSubject, numTrials);
choice_optimizing = zeros(numSubject, numTrials);

for i = 1:numSubject
    tem = num(i, 2:end);            % get rid of the ID
    data = tem(2:end);              % get rid of index of targets
    
    if tem(1) == 1                   % choose A if the reward bigger than 0
        data(find(data>0)) = 1;     % choosing A
        data(find(data<0)) = 0;     % choosing B
        data_max = data;
        data_max(find(data==0)) = 1;
        data_max(find(data==1)) = 0;       
    else
        data(find(data>0)) = 0;     % choosing B
        data(find(data<0)) = 1;     % choosing A
        data_max = data;
        data_max(find(data==1)) = 0;
        data_max(find(data==0)) = 1;       
        
    end
    choice_optimizing(i, :) = data_max;
    data = [rand(1, win)>0.5,data];
    for j = 1:numTrials
        probA = sum(data(j:j+win))/win;
        if getScoreA(probA)>getScoreB(1-probA)
            if data(win+j) == 1  % matching: chooseing A
                choice_matching(i, j) = 1; 
            end
        else
            if data(win+j) == 0  % matching: chooseing B
                choice_matching(i, j) = 1; 
            end            
        end
    end
end


