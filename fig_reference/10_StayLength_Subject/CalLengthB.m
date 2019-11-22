function numLength = CalLengthB(fileName, len)

data = xlsread(fileName);    % read experimental data from a file
numSubject = size(data, 1);  % the number of subjects
%numTrial = size(data, 2);
numLength = zeros(1, len);

for i = 1:numSubject
    tem = data(i, 2:end);           % get rid of the ID
    if tem(1) == 1                  % target A if the score is greater than 0
        tem = tem(1, 2:end);        % get rid of the ID for A or B
        indB = find(tem < 0, 1);
        it = 1;
        while indB > 0
            wind = findA(tem, indB, it, len);
            if wind > 0 && indB+wind<=len
                numLength(1, wind) = numLength(1, wind) + 1;
                tem(1:indB+wind) = [];
                indB = find(tem < 0, 1);
            else 
                indB = 0;
            end
        end
    else
        tem = tem(1, 2:end);        % get rid of the ID for A or B
        indB = find(tem > 0, 1);
        it = 0;
        while indB > 0
            wind = findA(tem, indB, it, len);
            if wind > 0 && indB+wind<=len
                numLength(1, wind) = numLength(1, wind) + 1;
                tem(1:indB+wind) = [];
                indB = find(tem > 0, 1);
            else 
                indB = 0;
            end
        end           
    end
    
end

function wind = findA(tem, indB, it, len)
% 找到indB之后的第一个A

tem(1:indB) = [];
if it == 1
    wind = find(tem > 0, 1);
    if length(wind) == 0 %&& length(tem) <= len % empty or more than len
        wind = 0;
%     elseif length(wind) == 0 && length(tem) >= len
%         wind = len;
    end 
else
    wind = find(tem < 0, 1);
    if length(wind) == 0 %&& length(tem) <= len % empty
        wind = 0;
%     elseif length(wind) == 0 && length(tem) >= len
%         wind = len;
    end
end
