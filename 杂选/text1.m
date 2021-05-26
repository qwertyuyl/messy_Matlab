clear
clc
m = [25 24 23 22 21];
D_m = [10.170,9.904,9.686,9.453,9.231];

n = [15 14 13 12 11];
D_n = [7.778 7.492 7.206 6.859 6.532];

lambda = 589.3*(10^(-9));

l1 = [21.251 21.248 21.255 21.249 21.253];
l2 = [3.118 3.110 3.114 3.115 3.112];

R1 = (D_m.*(10^(-3))).^2 - (D_n.*(10^(-3))).^2;
r2 = mean(R1)
r3 = r2/(4*(10)*lambda)

l1 = l1.*(10^(-3));
l2 = l2.*(10^(-3));
S_l = std(l1);
S_L = std(l2);
tp = 1.14;

U1 = sqrt((((tp/sqrt(5))*S_l)^2)+(0.002^2))
U2 = sqrt((((tp/sqrt(5))*S_L)^2)+(0.002^2))
Ud = sqrt(((U1/mean(l1))^2)+((U2/mean(l2)))^2)

d = 10*mean(l1)*lambda/(mean(l2))