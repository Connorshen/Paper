function numLength = CalLengthB(data, len)


numSubject = size(data, 1);  % the number of subjects
%numTrial = size(data, 2);
numLength = zeros(1, len);

for i = 1:numSubject
    tem = data(i, :);           % get rid of the ID
   
    indB = find(tem < 0, 1);

    while indB > 0
        wind = findA(tem, indB, len);
        if wind > 0 && indB+wind<=len
            numLength(1, wind) = numLength(1, wind) + 1;
            tem(1:indB+wind) = [];
            indB = find(tem < 0, 1);
        else 
            indB = 0;
        end
    end
    
    
end

function wind = findA(tem, indB, len)
% 找到indB之后的第一个A

tem(1:indB) = [];

wind = find(tem > 0, 1);
if length(wind) == 0 %&& length(tem) <= len % empty or more than len
    wind = 0;
end 

