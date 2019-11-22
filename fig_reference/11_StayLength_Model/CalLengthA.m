function numLength = CalLengthA(data, len)


numSubject = size(data, 1);  % the number of subjects
numLength = zeros(1, len);

for i = 1:numSubject
    tem = data(i, :);           % get rid of the ID

    indA = find(tem > 0, 1);    % find the first A

    while indA > 0
        wind = findA(tem, indA, len);   % find the first B
        if wind > 0 && indA+wind<=len
            numLength(1, wind) = numLength(1, wind) + 1;
            tem(1:indA+wind) = [];
            indA = find(tem > 0, 1);
        else 
            indA = 0;
        end
    end
end

function wind = findA(tem, indA, len)
% 找到indA之后的第一个B

tem(1:indA) = [];

wind = find(tem < 0, 1);
if length(wind) == 0 %&& length(tem) <= len % empty or more than len
    wind = 0;
end 

