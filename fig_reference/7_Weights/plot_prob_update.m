% plot the probability of updating synapses

[syn_A_begin_list, syn_B_begin_list, syn_A_end_list, syn_B_end_list, p_mch_update_list] = matching();

nSubjects = size(p_mch_update_list, 1);
nTrails = size(p_mch_update_list, 2);
x = 1:nTrails;

meanX_match = mean(p_mch_update_list);
errorX_match = std(p_mch_update_list)/sqrt(nSubjects); %error bar is standard error

[syn_A_begin_list, syn_B_begin_list, syn_A_end_list, syn_B_end_list, p_max_update_list] = maxmizing();
meanX_max = mean(p_max_update_list);
errorX_max = std(p_max_update_list)/sqrt(nSubjects); %error bar is standard error

figure;
hold on;
plot(x, meanX_match,'b', x, meanX_max, 'g');
legend('Attention to initinite reward', 'Attention to difference of reward', 'FontSize', 16);
shadedErrorBar(x, meanX_match, errorX_match, {'b-','markerfacecolor','b'});
shadedErrorBar(x, meanX_max, errorX_max, {'g-','markerfacecolor','g'});
hold off;
set(gca,'XTick',[0:100:500], 'YTick',[0:0.2:1]');
ylabel('Propotion of Updating Synapses', 'FontSize', 16);
xlabel('Trials', 'FontSize', 16);

set(gcf, 'renderer', 'painters');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [6 4]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 6 4]);
print(gcf, '-depsc2', 'prob_updating.eps');