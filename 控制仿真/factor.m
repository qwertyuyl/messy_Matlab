function f=factor(n)
if n==1
    f = 1;
    return;
else
    f = n*factor(n-1);
    return;
end
end