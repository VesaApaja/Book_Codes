#
# The Bisection Algorithm
# for generation of a brownian bridge between fixed end points.
#
import numpy as np
import matplotlib.pyplot as plt


# mid-points at stage i
def xmid(i):
    return ((M-i)*xs[i-1] + xp)/(M-i+1)

# tau at stage i
def tau_i(i):
    return (M-i)/(M-i+1)*tau


tau = 0.1
for ii in range(2, 5):
    M = 2**ii-1
    for i in range(M):
        print (M, tau_i(i)/tau)

exit()

# lamb = hbar^2/(2m) in some units
lamb = 0.5
# Number of imaginary time slices
# The number of sampled points is 2^i-1 for i=1,2,3...; point 0 and M are fixed
# => M = 2^i for i=1,2,3 ...; choose i = 6, M = 2^6
M = 2**3 
# inverse temperature
beta = 1.0
tau = beta/M 
# Fixed end points
x = 0.0
xp = 4.0

# sampled points
xs = np.zeros((M+1))
  
# Sample points and plot 

# points in bisection order
# Points are indexed 0, 1, 2, 3, ... M , with then 0 and M are fixed end points,
# and the sampled points are 1, 2, 3, ... M-1.
# Use a recursive function to list indices, together with the left and right known points

def bisection_indices(start, end):
    if end - start <= 1:
        return {}  
    mid = (start + end) // 2
    result = {mid: (start, end)}  # midpoint and its left and right edges
    # Recursively process left and right subranges
    result.update(bisection_indices(start, mid))
    result.update(bisection_indices(mid, end))
    return result

ivals = bisection_indices(0, M)
print('sampling order:')
for i, (left,right) in ivals.items():
    print(f'point {i} sampled using points {left} and {right}')

    
# tau values of all points
taus = [i*tau for i in range(M+1)]
# add fixed end points
xs[0] = x
xs[-1] = xp

for rep in range(5):
    # Sample points using bisection 
    for i, (left, right) in ivals.items():
        # tau-distance to left and right known points
        tau_i = 0.5*(taus[left]+taus[right])
        # deterministic midpoint
        xmid_i = 0.5*(xs[left] + xs[right])
        eta = np.random.normal()
        xs[i] = xmid_i + np.sqrt(2*lamb*tau_i)*eta
    
    plt.plot(xs,range(M+1),'o-')

plt.title(r"$\beta$ = "+str(beta))
plt.plot(x,0,'bo', markersize=10)
plt.plot(xp,M,'bo',  markersize=10)
plt.ylabel('imaginary time slice')
plt.xlabel('x coordinate')
plt.show()
