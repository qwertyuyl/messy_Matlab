function a = test1(b,c)
q = cond(b)
[w,e] = eig(c);
a = w*q
end