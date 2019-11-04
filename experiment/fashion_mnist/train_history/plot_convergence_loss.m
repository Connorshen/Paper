function plot_convergence_loss()
load("compare_convergence.mat")
len = size(compare_convergence,1);
rl_loss_all = [];
batch_loss_all = [];
for i = 1:len
    step_all = compare_convergence{i,1}(:,1);
    step_index = step_all(compare_convergence{i,1}(:,5)~=0);
    rl_loss = compare_convergence{i,1}(:,5);
    rl_loss = rl_loss(step_index)';
    rl_loss_all = [rl_loss_all;rl_loss];
    batch_loss = compare_convergence{i,2}(:,5);
    batch_loss = batch_loss(step_index)';
    batch_loss_all = [batch_loss_all;batch_loss];
end
rl_loss_all = mean(rl_loss_all,1);
batch_loss_all = mean(batch_loss_all,1);
plot(step_index,rl_loss_all,"r",step_index,batch_loss_all,"b");
legend("rl loss","rl batch loss");
xlabel("step");
ylabel("loss");
axis([-inf,inf,0,1])