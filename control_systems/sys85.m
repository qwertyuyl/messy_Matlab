function de = sys85(t,e)
de1 = e(2);
if(e(1)<-2)
    de2 = -2-e(1);
elseif(abs(e(1))<2)
    de2 = 0;
else de2 = 2-e(1);
end
de = [de1 de2]';
end