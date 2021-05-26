function dc = sys818(t,c)
dc1 = c(2);
if((c(1)>0.04)||(((c(1)<0.04)&&(c(1)>-0.04))&&(c(2)<0)))
    dc2 = -c(2)-2;
else dc2 = -c(2)+2;
end
dc = [dc1 dc2]';
end
