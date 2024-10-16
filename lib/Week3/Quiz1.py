import numpy as np
import matplotlib.pyplot as plt

r = 2.0 #ft

L = 5.0 #ft

V =  50 #ft³

def res(h):
    return V-L*(r**2*np.acos((r-h)/r)-(r-h)*np.sqrt(2*r*h-h**2))

def fn_bisection(xmin, xmax, mypoly): # BISECTION METHOD
    tol = 1e-10
    xf = 0
    yf = 0
    xa = (xmin + xmax)/2
    ya = mypoly(xa)
    if (abs(ya) < tol):
        xf = xa
        yf = ya
        return [xf, yf]
    if (ya * mypoly(xmin) < 0):
        xmax = xa
    else:
        xmin = xa
    [xf, yf] = fn_bisection(xmin, xmax, mypoly)
    return [xf, yf]

hrange = np.linspace(0, 2*r, 100)

[xf, yf] = fn_bisection(0, 2*r, res)

print(xf, yf)

plt.figure()
plt.grid()
plt.plot(hrange, res(hrange))
plt.plot(xf, yf, 'o')
plt.show()