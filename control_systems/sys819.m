function dc = sys819(t,c)
dc1 = c(2);
dc2 = c(3);
if((-2*c(1))<0)
    y = -1;
else y = 1;
end
dc3 = -4*c(2)-4*c(3)+5*y;
dc = [dc1 dc2 dc3]';
end
