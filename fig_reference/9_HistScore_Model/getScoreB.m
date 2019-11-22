function yB = getScoreB(x)
% 计算选择目标B的分值 
% ind : 选择目标B的比率
% yB : 分值
% scale = 240;
% x = ind*20;
% b = -1.05;
% r = 0.4;
% yB = -(1 ./ (1+exp(- r .* x)) + b - 0.15)*scale;
% %yB = -100.*ind+200;
  p1 = 265.7;
  p2 = -816.72;
  p3 = 940.97;
  p4 = -486.6;
  p5 = 144.84;
  
  yB = p1*x.^4 + p2*x.^3 +p3*x.^2 + p4*x +p5 ;