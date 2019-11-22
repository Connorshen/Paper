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

subplot(1, 3, 1);
hist(probA(:, 1))
h = findobj(gca,'Type','patch');
set(h,'FaceColor','blue','EdgeColor',[0 0 0]);
axis([0, 1, 0, 30]);
set(gca, 'XTick', 0:0.2:1)
set(gca, 'YTick', 0:5:25)
title('Experiment 1', 'FontSize', fontsize);
xlabel('Ratio of choice A', 'FontSize', fontsize);
ylabel('Number of subjects', 'FontSize', fontsize);

%set(gca, 'FontWeight', 'bold');


subplot(1, 3, 2);
hist(probA(:, 2));
h = findobj(gca,'Type','patch');
set(h,'FaceColor','blue','EdgeColor',[0 0 0]);
axis([0, 1, 0, 30]);
set(gca, 'XTick', 0:0.2:1)
set(gca, 'YTick', 0:5:25)
title('Experiment 2', 'FontSize', fontsize);
xlabel('Ratio of choice A', 'FontSize', fontsize);

%set(gca, 'FontWeight', 'bold');


subplot(1, 3, 3);
hist(probA(:, 3));
h = findobj(gca,'Type','patch');
set(h,'FaceColor','blue','EdgeColor',[0 0 0]);
axis([0, 1, 0, 30]);
set(gca, 'XTick', 0:0.2:1)
set(gca, 'YTick', 0:5:25)
title('Experiment 3', 'FontSize', fontsize);
xlabel('Ratio of choice A', 'FontSize', fontsize);

set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [12 3]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 12 3]);

print(gcf, '-depsc2', 'choiceDistribution.eps');

