function plot_convergence_acc(file_name)
load(file_name);
step = check_points(:,1);
% verify_acc
verify_acc = check_points(:,4);
verify_step = step(verify_acc~=0);
verify_acc = verify_acc(verify_acc~=0);
% train_acc
train_interval = 1000;
train_len = int32(size(check_points,1));
n_batch = idivide(train_len,train_interval,"ceil");
train_step = zeros(n_batch,1);
train_acc = zeros(n_batch,1);
for i=1:n_batch
    start_index = (i-1)*train_interval+1;
    end_index = start_index+train_interval-1;
    if end_index > train_len
        end_index = train_len;
    end
    train_acc(i) = mean(check_points(start_index:end_index,2));
    train_step(i) = end_index;
end

% plot(verify_step,verify_acc,train_step,train_acc)
plot(verify_step,verify_acc,'r',train_step,train_acc,'b')
legend("verify acc","train acc")
axis([-inf,inf,0,1])