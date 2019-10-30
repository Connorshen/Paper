clc
clear
load("train_result");
train_result = train_result(1:5999,:);
acc_all = [];
for i=1:60:5940
    acc = train_result(i:i+59,1);
    acc_all = [acc_all;mean(acc)];
end
plot(linspace(1,99,99),acc_all);
