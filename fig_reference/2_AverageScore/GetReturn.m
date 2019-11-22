function [return1, return2] = GetReturn()
% According the score function to calculate the return from two target
% Created by Stephen Z. Cheng, 2009,10,10

x = [1:1:20];
t = 0.215;
r = 0.4;
b1 = -1.05;
b2 = -1.05;
scale = 240;

y1 = t .* sqrt(x) + b1;
y2 = 1 ./ (1+exp(- r .* x)) + b2 - 0.15;

return1 = scale * -y1;
return2 = scale * -y2;