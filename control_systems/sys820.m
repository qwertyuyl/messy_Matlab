function de = sys820(t,e);
de1 = e(2);
if(e(1)<-1)
    de2 = -e(1);
elseif (abs(e(1))<1)
    de2 = -e(1) - e(2);
else 
    de2 = -e(1);
end
de = [de1 de2]';
end