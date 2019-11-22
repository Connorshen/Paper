function [score] = GetResponseReward(fileName)

num = xlsread(fileName);    % read experimental data from a file

numSubject = size(num, 1);          % the number of subjects
%numTrials = size(num, 2) - 2;       % the number of trials for every subject

scoreA = zeros(numSubject, 1);
scoreAll = zeros(numSubject, 1);

for i = 1:numSubject
    tem = num(i, 2:end);            % get rid of the ID
    if tem(1) == 1
        data = tem(1, 2:end);       % get rid of the ID for A or B
        sA = sum(data(find(data > 0)));
    else
        data = tem(1, 2:end);       % get rid of the ID for A or B
        sA = sum(data(find(data < 0)))*-1;
    end
    data(find(data<0)) = data(find(data<0))*-1;
    scoreA(i, 1) = sA;
    scoreAll(i, 1) = sum(data); 
end
score = [scoreA, scoreAll];