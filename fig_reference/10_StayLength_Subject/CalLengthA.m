function numLength = CalLengthA(fileName, len)

data = xlsread(fileName);    % read experimental data from a file
numSubject = size(data, 1);  % the number of subjects
numLength = zeros(1, len);

for i = 1:numSubject
    tem = data(i, 2:end);           % get rid of the ID
    if tem(1) == 1                  % target A if the score is greater than 0
        tem = tem(1, 2:end);        % get rid of the ID for A or B
        indA = find(tem > 0, 1);    % find the first A
        it = 1;
        while indA > 0
            wind = findA(tem, indA, it, len);   % find the first B
            if wind > 0 && indA+wind<=len
                numLength(1, wind) = numLength(1, wind) + 1;
                tem(1:indA+wind) = [];
                indA = find(tem > 0, 1);
            else 
                indA = 0;
            end
        end
    else
        tem = tem(1, 2:end);        % get rid of the ID for A or B
        indA = find(tem < 0, 1);
        it = 0;
        while indA > 0
            wind = findA(tem, indA, it, len);
            if wind > 0 && indA+wind<=len
                numLength(1, wind) = numLength(1, wind) + 1;
                tem(1:indA+wind) = [];
                indA = find(tem < 0, 1);
            else 
                indA = 0;
            end
        end           
    end
    
end

function wind = findA(tem, indA, it, len)
% 找到indA之后的第一个B

tem(1:indA) = [];
if it == 1
    wind = find(tem < 0, 1);
    if length(wind) == 0 %&& length(tem) <= len % empty or more than len
        wind = 0;
    end 
else
    wind = find(tem > 0, 1);
    if length(wind) == 0 %&& length(tem) <= len % empty
        wind = 0;
    end
end
