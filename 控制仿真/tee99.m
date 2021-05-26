clear
clc
clf

le = [];
x = '\xi=0.7,\omega=';
w = [2:2:12];
kesai = 0.7;
hold on
for Wn = w
    num = Wn^2;
    den = [1,2*kesai*Wn,Wn^2];
    step(num,den,6)
    
    num2str(Wn);
    y = [x,num2str(Wn)];
    le = [le,string(y)];
    
end
legend(le)