import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, exp, pi, sin, cos

D = 0.5
tau = 0.05
x = 0.2

def orig(xp,x,tau):
    r = 0.0
    for n in range(1,1000):
        term = 2*sin(n*pi*xp)*sin(n*pi*x)*exp(-tau*D*pi**2*n**2)
        r += term
        if max(abs(term))>1e-100:
            print(n,max(abs(term)))
    return r



def f1(xp,x,n,tau):
    return 1/sqrt(4*pi*D*tau)* exp(-(xp-x+2*n)**2/(4*D*tau))   

def f2(xp,x,n,tau):
    return 1/sqrt(4*pi*D*tau)* exp(-(xp+x+2*n)**2/(4*D*tau))   

xps = np.linspace(0,1,100)


plt.xlim([xps[0],xps[-1]])
plt.xlabel('x')
origs = orig(xps,x,tau)
plt.plot(xps,origs,'g-',label='original sin*sin*exp sum',lw=4.)



fsum = 0.0
f1s = np.zeros(len(xps))
f2s = np.zeros(len(xps))

for m in range(100):
    if m==0:
        ran = [0]
    else:
        ran = [-m,m]
    print('adding n = ',ran)
    for n in ran:
        f1s += f1(xps,x,n,tau)
        f2s += f2(xps,x,n,tau)
        
fsum += f1s-f2s        
plt.plot(xps,f1s,'r--',label='positive terms')
plt.plot(xps,-f2s,'b--',label='negative terms')
plt.plot(xps,fsum,'mx',label='positive + negative terms', markersize=4.)

# largest terms, accurate enough if tau is very small
def f1_red(xp,x,tau):
    return 1/sqrt(4*pi*D*tau)* exp(-(xp-x)**2/(4*D*tau))   

def f2_red(xp,x,tau):
    a = 1/sqrt(4*pi*D*tau)* exp(-(xp+x)**2/(4*D*tau))
    b = 1/sqrt(4*pi*D*tau)* exp(-(xp+x-2)**2/(4*D*tau))  
    #res = np.where(x<0.5, a, b)
    return a+b

xps2 = np.linspace(0,1,10)

f1s_reduced = f1_red(xps2,x,tau) 
plt.plot(xps2,f1s_reduced,'r*')
f2s_reduced = f2_red(xps2,x,tau) 
plt.plot(xps2,-f2s_reduced,'b*')


plt.legend()
plt.draw()
plt.pause(1e-3)
acc = max(abs(fsum-origs))
print('|max deviation| = ',acc)

    
plt.show()
