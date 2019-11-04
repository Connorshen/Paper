function plot_convergence_acc()
load("compare_convergence.mat")
len = size(compare_convergence,1);
rl_acc_all = [];
batch_acc_all = [];
for i = 1:len
    step_all = compare_convergence{i,1}(:,1);
    step_index = step_all(compare_convergence{i,1}(:,4)~=0);
    rl_acc = compare_convergence{i,1}(:,4);
    rl_acc = rl_acc(step_index)';
    rl_acc_all = [rl_acc_all;rl_acc];
    batch_acc = compare_convergence{i,2}(:,4);
    batch_acc = batch_acc(step_index)';
    batch_acc_all = [batch_acc_all;batch_acc];
end
rl_acc_all = mean(rl_acc_all,1);
batch_acc_all = mean(batch_acc_all,1);
plot(step_index,rl_acc_all,"r",step_index,batch_acc_all,"b");
legend("rl acc","rl batch acc");
xlabel("step");
ylabel("acc");
axis([-inf,inf,0,1])