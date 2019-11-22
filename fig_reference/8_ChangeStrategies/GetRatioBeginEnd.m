function [beginResponse, endResponse] = GetRatioBeginEnd(fileName, interval)

num = xlsread(fileName);    % read experimental data from a file

numSubject = size(num, 1);          % the number of subjects
snum = 10;

beginResponse  = zeros(numSubject, 1);      % the ratio of choosing target A
endResponse = zeros(numSubject, 1);   % mean score on every subject

for i = 1:numSubject
    tem = num(i, 2:end);            % get rid of the ID
    if tem(1) == 1
        data = tem(1, 2:end);       % get rid of the ID for A or B
        bdata = data(1, snum:interval+snum);
        edata = data(1, end-interval:end);
        
        bnumA = size(find(bdata > 0), 2);
        enumA = size(find(edata > 0), 2);
    else
        data = tem(1, 2:end);       % get rid of the ID for A or B
        bdata = data(1, snum:interval+snum);
        edata = data(1, end-interval:end);
        
        bnumA = size(find(bdata > 0), 2);
        enumA = size(find(edata > 0), 2);
    end

    beginResponse(i, 1) = bnumA/interval;
    endResponse(i, 1) = enumA/interval; 
end