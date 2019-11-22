% plot the weights before training and end training

clear all;
close all;

[syn_A_begin_list, syn_B_begin_list, syn_A_end_list, syn_B_end_list] = maxmizing();

nSubjects = size(syn_A_begin_list, 1);
xinterv = 0:0.01:0.1;
for n = 1:nSubjects
    barval_syn_A_begin(n, :) = hist(syn_A_begin_list(n, :)', xinterv);
    barval_syn_B_begin(n, :) = hist(syn_B_begin_list(n, :)', xinterv);
    barval_syn_A_end(n, :) = hist(syn_A_end_list(n, :)', xinterv);
    barval_syn_B_end(n, :) = hist(syn_B_end_list(n, :)', xinterv);
end
syn_A_begin_barval = mean(barval_syn_A_begin);
syn_A_begin_errors = std(barval_syn_A_begin);
mean_syn_A_begin = sum(xinterv .* syn_A_begin_barval) / sum(syn_A_begin_barval);

syn_B_begin_barval = mean(barval_syn_B_begin);
syn_B_begin_errors = std(barval_syn_B_begin);
mean_syn_B_begin = sum(xinterv .* syn_B_begin_barval) / sum(syn_B_begin_barval);

syn_A_end_barval = mean(barval_syn_A_end);
syn_A_end_errors = std(barval_syn_A_end);
mean_syn_A_end = sum(xinterv .* syn_A_end_barval) / sum(syn_A_end_barval);

syn_B_end_barval = mean(barval_syn_B_end);
syn_B_end_errors = std(barval_syn_B_end);
mean_syn_B_end = sum(xinterv .* syn_B_end_barval) / sum(syn_B_end_barval);

subplot(2, 2, 1)
mybar = bar(xinterv,syn_A_begin_barval ,'style', 'hist');
set(mybar, 'FaceColor',[0.952941179275513 0.87058824300766 0.733333349227905]);
hold on;
errorbar(xinterv, syn_A_begin_barval, syn_A_begin_errors, 'LineStyle','none','LineWidth',2,'Color',[0 0 0] );
plot([mean_syn_A_begin, mean_syn_A_begin], [0, 200], 'r--');
hold off;
set(gca,'XTick',[-0.02:0.02:0.12]','XTickLabel',num2str([-0.02:0.02:0.12]'), 'YLim', [0 200], 'YTick',[0:50:200]');
ylabel('Number of Weights', 'FontSize', 14);
title('The first trial connecting to A','FontSize', 14);

subplot(2, 2, 2)
mybar = bar(xinterv,syn_B_begin_barval ,'style', 'hist');
set(mybar, 'FaceColor',[0.952941179275513 0.87058824300766 0.733333349227905]);
hold on;
errorbar(xinterv, syn_B_begin_barval, syn_B_begin_errors, 'LineStyle','none','LineWidth',2,'Color',[0 0 0] );
plot([mean_syn_B_begin, mean_syn_B_begin], [0, 200], 'r--');
hold off;
set(gca,'XTick',[-0.02:0.02:0.12]','XTickLabel',num2str([-0.02:0.02:0.12]'), 'YLim', [0 200], 'YTick',[0:50:200]');
title('The first trial connecting to B','FontSize', 14);

subplot(2, 2, 3)
mybar = bar(xinterv,syn_A_end_barval ,'style', 'hist');
set(mybar, 'FaceColor',[0.756862759590149 0.866666674613953 0.776470601558685]);
hold on;
errorbar(xinterv, syn_A_end_barval, syn_A_end_errors, 'LineStyle','none','LineWidth',2,'Color',[0 0 0] );
plot([mean_syn_A_end, mean_syn_A_end], [0, 600], 'r--');
hold off;
set(gca,'XTick',[-0.02:0.02:0.12]','XTickLabel',num2str([-0.02:0.02:0.12]'), 'YLim', [0 600], 'YTick',[0:100:600]');
ylabel('Number of Weights', 'FontSize', 14);
xlabel('Weights', 'FontSize', 14);
title('The last trial connecting to A','FontSize', 14);

subplot(2, 2, 4)
mybar = bar(xinterv,syn_B_end_barval ,'style', 'hist');
set(mybar, 'FaceColor',[0.756862759590149 0.866666674613953 0.776470601558685]);
hold on;
errorbar(xinterv, syn_B_end_barval, syn_B_end_errors, 'LineStyle','none','LineWidth',2,'Color',[0 0 0] );
plot([mean_syn_B_end, mean_syn_B_end], [0, 600], 'r--');
hold off;
set(gca,'XTick',[-0.02:0.02:0.12]','XTickLabel',num2str([-0.02:0.02:0.12]'), 'YLim', [0 600], 'YTick',[0:100:600]');
xlabel('Weights', 'FontSize', 14);
title('The last trial connecting to B','FontSize', 14);

set(gcf, 'renderer', 'painters');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [12 8]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 12 8]);

print(gcf, '-depsc2', 'Weights_max_Change.eps');
