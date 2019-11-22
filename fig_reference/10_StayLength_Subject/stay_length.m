% 如何衡量被试只根据局部的经验来调整策略？由于实验实际是局部收益与全局收益之间的冲突
% 第一与第二个实验被试如何知道全局收益的信息？实际他只有局部收益的信息。
% 选择B后，看被试坚持了多久。
% Created by Stephen Z. Cheng

clear all
close all

linewidth = 2;
fontsize = 14;
fontweight = 'bold';

numSubject = 47;
numExp = 3;

len = 20; % the number of lookfward trials  

for indExp = 1:numExp
    fileName = ['/Volumes/Harderware/StephenDoc/Research/MyPapers/PreparedDraft/Addiction/codes/Figs/Data/Score_Exp', num2str(indExp), '.xls'];
    strTitle = ['Experiment ', num2str(indExp)];
    
    numLengthA = CalLengthA(fileName, len);
    numLengthB = CalLengthB(fileName, len);
    
    subplot(2, 3, indExp);
    bar(numLengthA/sum(numLengthA));
    set(gca,'ylim',[0 0.65])
    set(gca, 'YTick', 0:.2:0.6);
    set(gca,'yTickLabel',{'0','20','40', '60'});
    set(gca,'xlim',[0 20]);
    set(gca, 'xTick', 0:5:20);
    xlabel('Number of forward trials', 'FontSize', fontsize);
    title(strTitle, 'FontSize', fontsize);
    set(gca, 'FontSize', fontsize);
    
    subplot(2, 3, indExp+3);
    bar(numLengthB/sum(numLengthB));
    set(gca,'ylim',[0 0.65])
    set(gca, 'YTick', 0:.2:0.6);
    set(gca,'yTickLabel',{'0','20','40', '60'}); 
    set(gca,'xlim',[0 20]);
    set(gca, 'xTick', 0:5:20);
    xlabel('Number of forward trials', 'FontSize', fontsize);
    %title(strTitle);
    set(gca, 'FontSize', fontsize);
     
end
subplot(2, 3, 1);
ylabel('Frequence', 'FontSize', fontsize);
subplot(2, 3, 4);
ylabel('Frequence', 'FontSize', fontsize);

set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [12 6]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 12 6]);

print(gcf, '-depsc2', 'forwardTrails_subject.eps');