"""
Charles Wszalek Week 1 Day 3 Practice Code
"""


import numpy as np
import matplotlib.pyplot as plt

def mypoly(x):
    return x**3 + 4*x**2 + 2*x - 5

def fn_cross_check(N, xb_old, mypoly, counter):
    xa = np.linspace(xb_old[0], xb_old[1], N)
    yini = mypoly(xa[0])
    for i in range(2, len(xa)):  # WHY IS THIS 2 -> 10
        x = xa[i]
        yfin = mypoly(x)
        counter+=1
        if(yfin*yini <= 0):
            xb_new = [xa[i-1], xa[i]]
            xf = np.mean(xb_new)
            yf = mypoly(xf)
            break
        else:
            yini=yfin
    return [xf, yf, xb_new, counter]


xmin = -2
xmax = 2
TOL = .005

N = 10
xtot = np.linspace(xmin, xmax, N)
xminmax = [xmin, xmax]
counter = 1
dx = (xmax-xmin)/N

for i in range(1, 100):
    [xf, yf,xminmax, counter] = fn_cross_check(N, xminmax, mypoly, counter)
    if (abs(yf) < TOL):
        break

print(yf)
print(xf)
print(counter)

plt.figure()
plt.plot(xtot, mypoly(xtot))
plt.plot(xf, yf, '*')
plt.grid()
plt.show()

