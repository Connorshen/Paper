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
    fileName = ['/Volumes/Harderware/StephenDoc/Research/MyPapers/PreparedDraft/Addiction/codes/Figs/Data/Score_Exp', num2str(indExp), '.xls'];

    [probA(:, indExp), meanScore(:, indExp)] = GetProMean(fileName);
end

num_mat_1 = size(find(probA(:, 1) > 0.7 & probA(:, 1) < 0.85), 1);
num_max_1 = size(find(probA(:, 1) < 0.15), 1);

num_mat_2 = size(find(probA(:, 2) > 0.7 & probA(:, 2) < 0.85), 1);
num_max_2 = size(find(probA(:, 2) < 0.15), 1);

num_mat_3 = size(find(probA(:, 3) > 0.7 & probA(:, 3) < 0.85), 1);
num_max_3 = size(find(probA(:, 3) < 0.15), 1);

num_bar = [num_mat_1, num_max_1; num_mat_2, num_max_2; num_mat_3, num_max_3];
num_bar = num_bar./numSubject;
mybar = bar(num_bar, 'group');
set(mybar(1),'FaceColor','blue');
set(mybar(2),'FaceColor','green');
set(gca,'ylim',[0 1]);
set(gca, 'YTick', 0:0.2:1);
set(gca,'XTickLabel',{'Exp. 1','Exp. 2','Exp. 3'},'FontSize', fontsize);
ylabel('Fraction of subjects','FontSize', fontsize);
legend('Matching', 'Optimizing','FontSize', fontsize);
set(gca, 'FontSize', fontsize);

set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [5 4]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 5 4]);

print(gcf, '-depsc2', 'mat_max.eps');