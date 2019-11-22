
clear all
close all

linewidth = 2;
fontsize = 16;
fontweight = 'bold';

x = [1:1:20];
t = 0.215;
r = 0.4;
b = -1.05;
scale = 240;

y1 = t .* sqrt(x) + b;
y2 = 1 ./ (1+exp(- r .* x)) + b - 0.15;

return1 = scale * -y1;
return2 = scale * -y2;

p = 0:1/19:1;

income1 = return1.*p;
income2 = return2.*p(end:-1:1);
income = income1 + income2;

figure; hold on;

plot(p, return1, '-b', p, return2, '-g', 'linewidth', linewidth);
plot(p, income, 'r--', 'linewidth', linewidth);

xlabel('Proportion of responses to A','FontSize', fontsize);
ylabel('Value of reward', 'FontSize', fontsize)
%set(gca, 'FontWeight', 'bold');
legend('Choice A', 'Choice B', 'Expected utility');
set(gca, 'XTick', 0:0.2:1)
set(gca, 'YTick', 0:40:220)
set(gca, 'FontSize', fontsize);

set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [5 4]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 5 4]);

%print(gcf, '-depsc2', 'rewardstructure.eps');

