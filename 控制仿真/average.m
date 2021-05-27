function y=average(x)
% 向量元素的平均值
% 语法：average(x)，其中x 为输入向量
% 当输入非向量时，给出错误信息
[m,n]=size(x)
if(~((m==1)|(n==1))|(m==1&n==1))
error('Input must be a vector') % 判断输入是否为向量
end
y=sum(x)/length(x); % 实际计算过程
end