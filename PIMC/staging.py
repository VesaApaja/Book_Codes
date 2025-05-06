#
# The Staging Algorithm
# for generation of a brownian bridge path between given end points
#
import numpy as np
import matplotlib.pyplot as plt


# mid-points at stage i
def xmid(i):
    return ((M-i)*xs[i-1] + xp)/(M-i+1)

# tau at stage i
def tau_i(i):
    return (M-i)/(M-i+1)*tau
   

# lamb = hbar^2/(2m) in some units
lamb = 0.5
# Number of imaginary time slices
M = 50
# inverse temperature
beta = 0.1
tau = beta/M 
# Fixed end points
x = 0.0
xp = 4.0

# sampled points
xs = np.zeros((M+1))


# Sample points and plot 


ivals = [i for i in range(M+1)]
for rep in range(5):
    # Sample point x1,x2 ... x_{M-1} using staging
    
    for i in range(1,M):
        eta = np.random.normal()
        xs[i] = xmid(i) + np.sqrt(2*lamb*tau_i(i))*eta
    # add fixed end points
    xs[0] = x
    xs[-1] = xp
    plt.plot(xs,ivals,'o-')

plt.title(r"$\beta$ = "+str(beta))
plt.plot(x,0,'bo', markersize=10)
plt.plot(xp,M,'bo',  markersize=10)
plt.ylabel('imaginary time slice')
plt.xlabel('x coordinate')
plt.show()
