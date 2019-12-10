function [acc_final,acc_std] = get_final_acc()
load("compare_final_acc")
n_trial = size(compare_final_acc,1);
acc_max_all = [];
for i = 1:n_trial
    step_all = compare_final_acc{i,1}(:,1);
    step_index = step_all(compare_final_acc{i,1}(:,4)~=0);
    acc_all = compare_final_acc{i,1}(:,4);
    acc_all = acc_all(step_index);
    acc_max = max(acc_all);
    acc_max_all = [acc_max_all;acc_max];
end
acc_final = roundn(max(acc_max_all),-5);
acc_std = roundn(std(acc_max_all),-5);