% plot the propotion vs gains for three probability of attention
clear all;
close all;

linewidth = 2;
fontsize = 16;
fontweight = 'bold';

p_attention = cell(1, 2);
p_attention{1} = 0.3;
p_attention{2} = [0.3, 0.8];

num = length(p_attention);

figure;
for i = 1:num
    [pA_list,reward_listA, reward_listB] = matching_maxmizing(p_attention{i});
    
    subplot(1, num, i);
    ht = hist(mean(pA_list'), 0:0.1:1)/size(pA_list, 1);
    bar(0:0.1:1, ht,'FaceColor',[0.501960813999176 0.501960813999176 0.501960813999176]);
    axis([-0.05, 1, 0, 1]);
    set(gca, 'XTick', 0:0.2:1);
    set(gca, 'YTick', 0:0.2:1);
    set(gca, 'box', 'on');
    
    xlabel('mean % A', 'FontSize', fontsize);
    ylabel('% subjects', 'FontSize', fontsize);
end

set(gcf, 'renderer', 'painters');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [10 4]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 4]);
print(gcf, '-depsc2', 'histscore_model.eps');