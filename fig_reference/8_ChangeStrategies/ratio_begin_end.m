% Response ratio to target A in begin 40 vs end 40
% Created by Stephen Z. Cheng

clear all;
close all;

linewidth = 2;
fontsize = 16;
fontweight = 'bold';

numSubject = 47;
interval = 30;
beginResponse = zeros(numSubject, 1);
endResponse = zeros(numSubject, 1);


numExp = 3;
figure; hold on;
for indExp = 1:numExp
    fileName = ['/Volumes/Harderware/StephenDoc/Research/MyPapers/PreparedDraft/Addiction/codes/Figs/Data/Score_Exp', num2str(indExp), '.xls'];

    [beginResponse(:, indExp), endResponse(:, indExp)] = GetRatioBeginEnd(fileName, interval);
    
    if indExp == 1
        plot(beginResponse(:, indExp), endResponse(:, indExp), '^', 'color', 'red','MarkerSize',10);
    elseif indExp == 2
        plot(beginResponse(:, indExp), endResponse(:, indExp), 'o', 'color', 'green','MarkerSize',10);    
    else
        plot(beginResponse(:, indExp), endResponse(:, indExp), '*', 'color', 'blue','MarkerSize',10);
    end
    

end
legend1 = legend('Exp. 1', 'Exp. 2','Exp. 3');
set(legend1,'Location','SouthEast');

plot([0, 1],[0, 1], '--black', 'Linewidth', linewidth);
axis([0, 1, 0, 1]);
set(gca, 'XTick', 0:0.2:1)
set(gca, 'YTick', 0:0.2:1)
set(gca, 'FontSize', fontsize)
set(gca, 'box', 'on');
xlabel('Fraction A in trials 10-40', 'FontSize', fontsize);

ylabel('Fraction A in trials 70-100', 'FontSize', fontsize);

set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [5 4]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 5 4]);

print(gcf, '-depsc2', 'ratio_begin_end.eps');