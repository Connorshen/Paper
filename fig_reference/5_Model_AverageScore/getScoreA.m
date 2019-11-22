function yA = getScoreA(x)
% 计算选择目标A的分值 
% ind : 选择目标A的比率
% yA : 分值
  p1 = 192.2;
  p2 = -507;
  p3 = 526.12;
  p4 = -388.75;
  p5 = 199.22;
  
  yA = p1*x.^4 + p2*x.^3 +p3*x.^2 + p4*x +p5 ;
% scale = 240;
% x = ind*20;
% b = -1.05;
% t = 0.215;
% yA = -(t .* sqrt(x) + b)*scale;
%yA = -100.*ind+250;