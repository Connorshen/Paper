function [choiceslide] = GetChoiceList(fileName)
% Created by Stephen Z. Cheng, 2009,10,11

data = xlsread(fileName);    % read experimental data from a file

numSubject = size(data, 1);          % the number of subjects
numTrails = 100;

data = data(:, 2:end);                 % get rid of the ID
data_ = data(:, 2:end);                % get rid of the ID for A or B
choice_list = zeros(size(data_));         % get rid of the ID for A or B

% %conv the choice
% tau = -4:4;
% n = 4;
% sigma = 2;
% w = 1/(sqrt(2*pi)*sigma)*exp(-tau.^2/(2*sigma^2));% gaussian filter function
% w = w';
slide_wind = 10;

for i = 1:numSubject

    if data(i, 1) == 1
        choice_list(i, find(data_(i, :) > 0)) = 1;
    else
        choice_list(i, find(data_(i, :) < 0)) = 1;
    end

%     t1 = conv(choice_list(i, :),w); 
%     choice(i, :) = t1(n+1:end-n);
    choice = [];
    for t = 1:numTrails/slide_wind
        choice = [choice, mean(choice_list(i, slide_wind*(t-1)+1 : slide_wind*t))];
    end
    choiceslide(i, :) = choice;
end


