function plot_class()
load("compare_class.mat")
n_classes = size(compare_class,1);
n_out_features = size(compare_class,2)/2;
n_categorys = [];
out_features_cpls = [];
acc_all = [];
for i=1:n_classes
    acc_class = [];
    for j=1:n_out_features
        init_para = compare_class{i,j*2};
        out_features_cpl = init_para.out_features_cpl;
        n_category = init_para.n_category;
        step_all = compare_class{i,j*2-1}(:,1);
        step_index = step_all(compare_class{i,j*2-1}(:,4)~=0);
        acc = compare_class{i,j*2-1}(:,4);
        acc = acc(step_index);
        acc = max(acc);
        acc_class = [acc_class;acc];
        if i==1
            out_features_cpls = [out_features_cpls;out_features_cpl];
        end
    end
    n_categorys = [n_categorys;n_category];
    acc_all = [acc_all;acc_class'];
end
figure(1)
set(gcf,"Position",[500,500,600,400], "color","w")
plot(1:size(n_categorys,1),acc_all);
axis([0.5,5.5,0,1.1])
set(gca,'XTick',1:size(n_categorys,1));
set(gca,'xticklabel',n_categorys);
xlabel("n category")
ylabel("acc")
legend(num2str(out_features_cpls),"Location","SouthWest");
title("compare category with cpl out features")