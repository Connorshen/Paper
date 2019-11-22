function r = getReward(action_list, i, j, iniAction)
% 根据奖励函数计算当前动作的奖励数
% 输入：action_list, 记录动作历史
%      i, 第 i 次选择
% 输出：奖励值
% Created by Stephen Z. Cheng

action = action_list(j, 1:i);
W = 20;

if i <= W
    numA = sum(action)+sum(iniAction(i+1:end));
    indA = numA/W; % 前W次动作中，选择按钮A的比率    
    if action(1, i) % 选择按钮A
        r = getScoreA(indA); % 根据选择按钮A的比率计算奖励值
        return;
    else            % 选择按钮B
        r = getScoreB(indA);
        return;
    end
else % W次后的选择
	numA = sum(action(end-W+1:end));
	indA = numA/W;  % 前W次动作选择按钮A的比率  
    if action(1, i) % 选择按钮A
        r = getScoreA(indA);
        return;
    else            % 选择按钮B   
        r = getScoreB(indA);
        return;
    end
end



