function dc = sys825(t,c)
dc1 = c(2);
dc2 = c(3);
if(c(1)<0)
    dc3 = 5-0.5*c(1)-0.5*c(2)-1.5*c(3);
else dc3 = -5-0.5*c(1)-0.5*c(2)-1.5*c(3);
end
dc = [dc1 dc2 dc3]';
end
