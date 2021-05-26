clear
x = []%x轴数据，维度为（*，1）
y = []%y轴数据，维度为（*，1）
x = reshape(x,1,13)%转换维度
y = reshape(y,1,13)%转换维度
[p,s] = polyfit(x,y,1);%计算回归参数
plot(x,y,'o');%绘出实际点位置
x0 = [0:0.1:max(x)+1];%回归方程x的取值
y0 = p(1)*x0+p(2);%回归方程的表达
hold on
plot(x0,y0)%绘出回归直线
title('Uh--Is关系图线');%标题题目
xlabel('Is（mA）');%x轴表示物理量
ylabel('Uh（mV）');%y轴表示物理量
p(1),p(2)