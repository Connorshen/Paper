function [probA, meanScore] = GetProMean(fileName)
% Created by Stephen Z. Cheng, 2009,10,11

num = xlsread(fileName);    % read experimental data from a file

numSubject = size(num, 1);          % the number of subjects
numTrials = size(num, 2) - 2;       % the number of trials for every subject

probA  = zeros(numSubject, 1);      % the ratio of choosing target A
meanScore = zeros(numSubject, 1);   % mean score on every subject

for i = 1:numSubject
    tem = num(i, 2:end);            % get rid of the ID
    if tem(1) == 1
        data = tem(1, 2:end);       % get rid of the ID for A or B
        numA = size(find(data > 0), 2);
    else
        data = tem(1, 2:end);       % get rid of the ID for A or B
        numA = size(find(data < 0), 2);        
    end
    data(find(data<0)) = data(find(data<0))*-1;
    meanScore(i, 1) = sum(data)/numTrials;
    probA(i, 1) = numA/numTrials; 

end