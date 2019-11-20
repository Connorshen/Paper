y=randn(30,80); %随机生成30行80列的数据
x=1:size(y,2); % x 为shadeErrorBar的横轴
shadedErrorBar(x,mean(y,1),std(y),'lineprops','g');%参数分别是，横轴刻度值，y的均值，y的标准差，配置线条的颜色为绿色