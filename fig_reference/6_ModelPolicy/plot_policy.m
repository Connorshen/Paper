% plot the policy obtained by model
clear all;
close all;

linewidth = 2;
fontsize = 16;
fontweight = 'bold';

load PMatch
p_match_model = pA_list;
clear pA_list;
load PMax
p_max_model = pA_list;

p_match = 0.76;
p_max   = 0;
numTrials = size(p_match_model,2);
x = 1:numTrials;

meanX_match = mean(p_match_model);
%errorX_match = std(p_match_model)/sqrt(size(p_match_model, 1)); %error bar is standard error
errorX_match = std(p_match_model);

meanX_max = mean(p_max_model);
%errorX_max = std(p_max_model)/sqrt(size(p_max_model, 1));
errorX_max = std(p_max_model);


figure;
hold on;
plot([0 numTrials], [p_match p_match], '--b', 'LineWidth', 2);
plot([0 numTrials], [p_max p_max], '--g', 'LineWidth', 2);
legend('Matching Policy', 'Maxmizing Polity' );

shadedErrorBar(x, meanX_match, errorX_match, {'b-','markerfacecolor','b'});

shadedErrorBar(x, meanX_max, errorX_max, {'g-','markerfacecolor','g'});
plot([0 numTrials], [p_match p_match], '--b', 'LineWidth', 2);
plot([0 numTrials], [p_max p_max], '--g', 'LineWidth', 2);

hold off;


set(gca, 'ylim', [0 1]);
set(gca, 'xlim', [1 numTrials]);
set(gca, 'yTick', [0:0.2:1]);
set(gca, 'xTick', [0:100:numTrials]);

xlabel('Trials', 'FontSize', fontsize);
ylabel('Propotion of Choosing Matching Button', 'FontSize', fontsize);
set(gca, 'FontSize', fontsize);
%set(gca, 'FontWeight', 'bold');

set(gcf, 'renderer', 'painters');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [6 4]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 6 4]);
print(gcf, '-depsc2', 'policy_model.eps');