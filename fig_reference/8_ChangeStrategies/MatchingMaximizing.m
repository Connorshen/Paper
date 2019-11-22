% Response ratio to target A in begin 40 vs end 40
% Created by Stephen Z. Cheng

clear all;
close all;

linewidth = 2;
fontsize = 16;
fontweight = 'bold';
numTrails = 100;
numSubject = 47;


matching = zeros(numSubject, numTrails);
maximizing = zeros(numSubject, numTrails);

numExp = 3;

win_begin = 1:40;
win_end = 50:100;

figure; 
subplot(1, 2, 1);hold on;
for indExp = 1:numExp
    fileName = ['/Volumes/Harderware/StephenDoc/Research/MyPapers/PreparedDraft/Addiction/codes/Figs/Data/Score_Exp', num2str(indExp), '.xls'];

    [matching, maximizing] = GetMatchMax(fileName);
    
    matching = matching';
    frac_matching = sum(matching(win_begin, :))/length(win_begin);
    
    maximizing = maximizing';
    frac_maximizing = sum(maximizing(win_begin, :))/length(win_begin);
    
    if indExp == 1
        plot(frac_matching, frac_maximizing, '^', 'color', 'red','MarkerSize',10);
    elseif indExp == 2
        plot(frac_matching, frac_maximizing,  'o', 'color', 'green','MarkerSize',10);    
    else
        plot(frac_matching, frac_maximizing,  '*', 'color', 'blue','MarkerSize',10);
    end
    
end
hold off;
legend('Exp.1', 'Exp.2', 'Exp.3');
axis([0, 1, 0, 1]);
title('Trials 1-40','FontSize', fontsize);
set(gca, 'XTick', 0:0.2:1)
set(gca, 'YTick', 0:0.2:1)
set(gca, 'box', 'on');
xlabel('% trials matching', 'FontSize', fontsize);
ylabel('% trials maximizing', 'FontSize', fontsize);

subplot(1, 2, 2);hold on;
for indExp = 1:numExp
    fileName = ['/Volumes/Harderware/StephenDoc/Research/MyPapers/PreparedDraft/Addiction/codes/Figs/Data/Score_Exp', num2str(indExp), '.xls'];

    [matching, maximizing] = GetMatchMax(fileName);
    
    matching = matching';
    frac_matching = sum(matching(win_end, :))/length(win_end);
    
    maximizing = maximizing';
    frac_maximizing = sum(maximizing(win_end, :))/length(win_end);
    
    if indExp == 1
        plot(frac_matching, frac_maximizing, '^', 'color', 'red','MarkerSize',10);
    elseif indExp == 2
        plot(frac_matching, frac_maximizing,  'o', 'color', 'green','MarkerSize',10);    
    else
        plot(frac_matching, frac_maximizing,  '*', 'color', 'blue','MarkerSize',10);
    end
    
end
legend('Exp.1', 'Exp.2', 'Exp.3');
axis([0, 1, 0, 1]);
title('Trials 50-100','FontSize', fontsize);
set(gca, 'XTick', 0:0.2:1)
set(gca, 'YTick', 0:0.2:1)
set(gca, 'box', 'on');
xlabel('% trials matching', 'FontSize', fontsize);


set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [10 4]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 4]);

print(gcf, '-depsc2', 'ratio_matching_maximizing.eps');