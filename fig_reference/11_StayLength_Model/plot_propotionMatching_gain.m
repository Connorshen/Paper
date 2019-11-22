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
len = 20; % the number of lookfward trials  

figure;
for i = 1:num
    [pA_list,reward_list] = matching_maxmizing(p_attention{i});
    

    numLengthA = CalLengthA(reward_list, len);
    numLengthB = CalLengthB(reward_list, len);
    
    subplot(2, 2, i);
    bar(numLengthA/sum(numLengthA));
    set(gca,'ylim',[0 0.65])
    set(gca, 'YTick', 0:.2:0.6);
    set(gca,'yTickLabel',{'0','20','40', '60'});
    set(gca,'xlim',[0 20]);
    set(gca, 'xTick', 0:5:20);
    xlabel('Number of forward trials', 'FontSize', fontsize);
    set(gca, 'FontSize', fontsize);
    
    subplot(2, 2, i+2);
    bar(numLengthB/sum(numLengthB));
    set(gca,'ylim',[0 0.65])
    set(gca, 'YTick', 0:.2:0.6);
    set(gca,'yTickLabel',{'0','20','40', '60'}); 
    set(gca,'xlim',[0 20]);
    set(gca, 'xTick', 0:5:20);
    xlabel('Number of forward trials', 'FontSize', fontsize);
    set(gca, 'FontSize', fontsize);
end
subplot(2, 2, 1);
ylabel('Frequence', 'FontSize', fontsize);
subplot(2, 2, 3);
ylabel('Frequence', 'FontSize', fontsize);

set(gcf, 'renderer', 'painters');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [10 6]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 6]);
print(gcf, '-depsc2', 'forwardTrails_model.eps');