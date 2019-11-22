% policy of subjects

clear all 
close all

linewidth = 2;
fontsize = 16;
fontweight = 'bold';

p_match = 0.76;
p_max   = 0;


numSubject = 47;
numExp = 3;
numTrials = 100;
probA = zeros(numSubject, numExp);
meanScore = zeros(size(probA));

numSubject = 47;
numExp = 3;

match_choiceList = [];
max_choiceList = [];



for indExp = 1:numExp
    fileName = ['/Volumes/Harderware/StephenDoc/Research/MyPapers/PreparedDraft/Addiction/codes/Figs/Data/Score_Exp', num2str(indExp), '.xls'];

    choiceList = GetChoiceList(fileName);
    [probA(:, indExp), meanScore(:, indExp)] = GetProMean(fileName);
    for n = 1:numSubject
        if probA(n, indExp) > 0.70 && probA(n, indExp) < 0.85 %matching
            match_choiceList = [match_choiceList; choiceList(n, :)];
            
        elseif probA(n, indExp) < 0.15  %optimize
            max_choiceList = [max_choiceList; choiceList(n, :)];
            
        end
    end
end
figure; hold on;

plot([0 numTrials], [p_match p_match], '--b', 'LineWidth', linewidth);
plot([0 numTrials], [p_max p_max], '--g', 'LineWidth', linewidth);
legend('Matching Policy', 'Maxmizing Polity' );

x = 10:10:100;
errorbar(x, mean(match_choiceList), std(match_choiceList), 'color', 'blue',  'LineWidth', linewidth);
errorbar(x, mean(max_choiceList), std(max_choiceList), 'color', 'green',  'LineWidth', linewidth);

hold off;

set(gca, 'ylim', [0 1.3]);
set(gca, 'xlim', [1 numTrials]);
set(gca, 'yTick', [0:0.2:1]);
set(gca, 'xTick', [0:20:numTrials]);

xlabel('Trials','FontSize', fontsize);
ylabel('Propotion of Chooce A','FontSize', fontsize);
set(gca, 'FontSize', fontsize);

set(gcf, 'renderer', 'painters');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [6 4]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 6 4]);
print(gcf, '-depsc2', 'policy_subjects.eps');

% numMatch = size(match_choiceList, 2);
% numMax = size(max_choiceList, 2);
% shadedErrorBar(1:numMatch, mean(match_choiceList), std(match_choiceList));
% shadedErrorBar(1:numMax, mean(max_choiceList), std(max_choiceList));