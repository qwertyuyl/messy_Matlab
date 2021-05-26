function [dx] = sys0001(t,x)
%º¯ÊıÃèÊö
% dx = 5*sin(x);dy=x+2y
    dx = [5*sin(x(1));
        x(1)+2*x(2)];

end
