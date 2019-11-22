% ??????????????????????????????A??????????????????
% Created by Stephen Z. Cheng

clear all 
close all

linewidth = 2;
fontsize = 16;
fontweight = 'bold';

numSubject = 47;
numExp = 3;
probA = zeros(numSubject, numExp);
meanScore = zeros(size(probA));

for indExp = 1:numExp
    fileName = ['../Data/Score_Exp', num2str(indExp), '.xls'];

    [probA(:, indExp), meanScore(:, indExp)] = GetProMean(fileName);
end

[return1, return2] = GetReturn(); % get expected rewards conditioned on target A or B is selected

p = 0:1/19:1;

income1 = return1.*p;
income2 = return2.*p(end:-1:1);
income = income1 + income2;     % expected rewards from two targets

subplot(1, 3, 1);
plot(p, return1, '-b', p, return2, '-g', 'linewidth', linewidth);
hold on;
plot(probA(:, 1), meanScore(:, 1), '.', 'Markersize', 15, 'color', 'black');
plot(p, income, 'r--','linewidth', linewidth);
set(gca, 'XTick', 0:0.2:1)
set(gca, 'YTick', 0:40:220)
title('Experiment 1','FontSize', fontsize);
xlabel('Ratio of choice A', 'FontSize', fontsize);
ylabel('Score Value', 'FontSize', fontsize);
set(gca, 'FontSize', fontsize);
box off;

subplot(1, 3, 2);
plot(p, return1, '-b', p, return2, '-g', 'linewidth', linewidth);
hold on;
plot(probA(:, 2), meanScore(:, 2), '.', 'Markersize', 15, 'color', 'black');
plot(p, income, 'r--','linewidth', linewidth);
set(gca, 'XTick', 0:0.2:1)
set(gca, 'YTick', 0:40:220)
title('Experiment 2','FontSize', fontsize);
xlabel('Ratio of  choice A', 'FontSize', fontsize);
set(gca, 'FontSize', fontsize);
box off;

subplot(1, 3, 3);
plot(p, return1, '-b', p, return2, '-g', 'linewidth', linewidth);
hold on;
plot(probA(:, 3), meanScore(:, 3), '.', 'Markersize', 15, 'color', 'black');
plot(p, income, 'r--','linewidth', linewidth);
set(gca, 'XTick', 0:0.2:1)
set(gca, 'YTick', 0:40:220)
set(gca, 'FontSize', fontsize);
title('Experiment 3', 'FontSize', fontsize);
xlabel('Ratio of  choice A', 'FontSize', fontsize);

box off;

set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [12 4]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 12 4]);

print(gcf, '-depsc2', 'averageScore.eps');
