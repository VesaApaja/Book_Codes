#
# Partition function paths in free diffusion
# 
# Pick a few 2D particle positions and use staging
# to find a sample of imaginary time paths

import numpy as np
import matplotlib.pyplot as plt

# mid-points at stage i
def xmid(i):
    return ((M-i)*xs[:,:,i-1] + xs[:,:,-1])/(M-i+1)

# tau at stage i
def tau_i(i):
    return (M-i)/(M-i+1)*tau

# lamb = hbar^2/(2m) in some units
lamb = 0.5
# Number of imaginary time slices
M = 10

# Number of particle
N = 6
# space dimension
dim = 2

# particle paths
xs = np.zeros((dim, N, M+1)) # to be filled
# particle position at end points
xs[:,:,0] = np.random.random((dim, N))
xs[:,:,M] = xs[:,:,0] # loop


# inverse temperature
for beta in [0.001, 0.01, 0.1]:
    tau = beta/M
    # sample paths
    for i in range(1,M):
        eta = np.random.normal(size=(dim,N))  #  vector
        xs[:,:,i] = xmid(i) + np.sqrt(2*lamb*tau_i(i))*eta
        
    plt.figure()
    plt.xlim([-0.5,1.5])
    plt.ylim([-0.5,1.5])
    for n in range(N):
        plt.title(r"$\beta$ = "+str(beta))
        path, = plt.plot(xs[0,n,0],xs[1,n,0],'o', markersize=10, markeredgecolor='black')
        col = path.get_color()
        plt.plot(xs[0,n,:],xs[1,n,:], 'o-', color=col)
        plt.draw()
        plt.pause(1e-3)

plt.show()    
