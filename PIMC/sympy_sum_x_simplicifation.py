from sympy import *

#m, n = symbols('m n', cls=Idx)
m = Symbol('m', integer=true)

x = IndexedBase('x')
t = IndexedBase('t')
M = Symbol('M', integer=true)     

lhs = Sum( (x[m]-x[1])*(-(x[m]-x[m+1])/t[m] + (x[m-1]-x[m])/t[m-1]), (m,1,M))
rhs = Sum(-(x[m]-x[m+1])**2/t[m], (m,1,M)) - (x[M+1]-x[1])*(x[M]-x[M+1])/t[M]

pprint(lhs)
pprint(rhs)
lhs_5=expand(lhs.subs(M,5).doit())
rhs_5=expand(rhs.subs(M,5).doit())

print('using M=5, compute the difference of the expressions:')
pprint(lhs_5-rhs_5)
