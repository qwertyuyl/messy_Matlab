function [c,ce] = cfun(x)
%约束minfun函数的约束条件
%   ~~~~
    ce = [];
    c = [x(1)+x(2);
        1.5+x(1)*x(2)-x(1)-x(2);
        -10-x(1)*x(2)]
end

