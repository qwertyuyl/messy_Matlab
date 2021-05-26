function [outputArg1,outputArg2] = pisure()
% 求取pi/4的近似值
%   利用pi/4 = 1-1/3+1/5-1/7...进行计算
i=1;
A = 1;
n = 0;
while abs(A-(pi/4))>10^(-6)
    i=i+2;
    n = n+1;
    A = A + (-1)^(n)*(1/i);
end
outputArg1 = A;
outputArg2 = i;
end

