% Response ratio to score ratio in target A for every subject
% Created by Stephen Z. Cheng

clear all;
close all;

linewidth = 2;
fontsize = 16;
fontweight = 'bold';

numSubject = 47;

numExp = 3;
for indExp = 1:numExp
    fileName = ['/Volumes/Harderware/StephenDoc/Research/MyPapers/PreparedDraft/Addiction/codes/Figs/Data/Score_Exp', num2str(indExp), '.xls'];
    strTitle = ['Experiment ', num2str(indExp)];
    [score] = GetResponseReward(fileName);
    if indExp == 1
        score_1 = score;
    elseif indExp == 2
        score_2 = score;
    else
        score_3 = score;
    end
    
    subplot(1, 3, indExp);
    hist(score(:, 2), 4000:1000:12000);
    h = findobj(gca,'Type','patch');
    set(h,'FaceColor',[0 0.5 0],'EdgeColor',[0 0 0]);
    set(gca, 'xTick', 4000:2000:12000);
    set(gca,'XTickLabel',{'4','6','8', '10', '12x10'});
    set(gca, 'ylim', [0 25]);
    set(gca, 'xlim', [3000 13000]);
    xlabel('Score', 'FontSize', fontsize);
    title(strTitle, 'FontSize', fontsize);
end
subplot(1, 3, 1);
ylabel('Number of subjects', 'FontSize', fontsize);

set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [12 3]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 12 3]);

print(gcf, '-depsc2', 'RewardDistribution.eps');

