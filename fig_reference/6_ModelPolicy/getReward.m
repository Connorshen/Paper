function r = getReward(action_list, i, j, iniAction)
% ���ݽ����������㵱ǰ�����Ľ�����
% ���룺action_list, ��¼������ʷ
%      i, �� i ��ѡ��
% ���������ֵ
% Created by Stephen Z. Cheng

action = action_list(j, 1:i);
W = 20;

if i <= W
    numA = sum(action)+sum(iniAction(i+1:end));
    indA = numA/W; % ǰW�ζ����У�ѡ��ťA�ı���    
    if action(1, i) % ѡ��ťA
        r = getScoreA(indA); % ����ѡ��ťA�ı��ʼ��㽱��ֵ
        return;
    else            % ѡ��ťB
        r = getScoreB(indA);
        return;
    end
else % W�κ��ѡ��
	numA = sum(action(end-W+1:end));
	indA = numA/W;  % ǰW�ζ���ѡ��ťA�ı���  
    if action(1, i) % ѡ��ťA
        r = getScoreA(indA);
        return;
    else            % ѡ��ťB   
        r = getScoreB(indA);
        return;
    end
end



