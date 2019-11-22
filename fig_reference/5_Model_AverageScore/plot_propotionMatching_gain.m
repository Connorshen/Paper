% plot the propotion vs gains for three probability of attention
clear all;
close all;

linewidth = 2;
fontsize = 16;
fontweight = 'bold';

p_attention = [0, 0.5, 1];
num = length(p_attention);

figure;
for i = 1:num
    [pA_list,reward_listA, reward_listB] = matching_maxmizing(p_attention(i));
    
    subplot(1, num, i);
    hold on;
    ind = 0:0.01:1;
    plot(ind, getScoreA(ind), 'b', 'LineWidth', linewidth);
    plot(ind, getScoreB(ind), 'g', 'LineWidth', linewidth);
    plot(mean(pA_list'), mean((reward_listA+reward_listB)'), 'o', 'Markersize', 6, 'MarkerEdgeColor' , 'none'  ,'MarkerFaceColor' , 'black');
    set(gca, 'XTick', 0:0.2:1);
    set(gca, 'YTick', 0:40:220);
    xlabel('Propotion of Choice A','FontSize', fontsize);
    if i == 1
        ylabel('Average Reward','FontSize', fontsize);
    end
    hold off;
end

set(gcf, 'renderer', 'painters');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [16 4]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 16 4]);
print(gcf, '-depsc2', 'attention_policy.eps');