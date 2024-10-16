"""
Charles Wszalek Assignment 1
"""
import numpy as np
import matplotlib.pyplot as plt
from lib.Header import PDF, SAVE, print2 # Custom library to replicate Publish in Matlab


def func1(x):
    return 2*x + 1


def func2(x):
    return x**2 + 2*x - 1


def func3(x):
    return x**3 + 4*x**2 + 2*x - 5


def func4(x):
    return x**5 + 10*x**3 + 4*x**2 + 2*x + 1


min = -2
max = 2
N = 100

xrange = np.zeros(N)
dx = (max-min)/N
xrange[0] = min
for i in range(len(xrange)-1):
    xrange[i+1] = xrange[i] + dx

plt.figure()
plt.grid()
plt.title('$2x + 1$', fontsize=14)
plt.plot(xrange, func1(xrange), linewidth=2)
SAVE(1)
plt.show()

plt.figure()
plt.grid()
plt.title('$x^2 + 2x - 1$', fontsize=14)
plt.plot(xrange, func2(xrange), linewidth=2)
SAVE(2)
plt.show()

plt.figure()
plt.grid()
plt.title('$x^3 + 4x^2 + 2x - 5$', fontsize=14)
plt.plot(xrange, func3(xrange), linewidth=2)
SAVE(3)
plt.show()

plt.figure()
plt.grid()
plt.title('$x^5 + 10x^3 + 4x^2 + 2x + 1$', fontsize=14)
plt.plot(xrange, func4(xrange), linewidth=2)
SAVE(4)
plt.show()

PDF("Assignment1.py", "Assignment1.pdf")


