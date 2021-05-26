function de = sys81(t,e)
J = 1;
K1 = 1;
K2 = 2;
de1 = e(2);
de2 = -(K1/(J+K1*K2))*e(1);
de = [de1 de2]';

end


