import numpy as np
import matplotlib.pyplot as plt
from lib.Header import *

def func(x):
    return 1 / (1 + x ** 4)

def Integrate(func, x1, x2, N):
    x = np.linspace(x1, x2, N + 1)
    y = func(x)
    area = 0
    for i in range(len(x)-1):
        area += (y[i] + y[i+1])/2 * (x2-x1)/N
    return area


N = [5, 10, 20, 50, 100, 1000]

xmin = 0
xmax = 3
A = np.zeros(6)

for i in range(len(N)):
    A[i] = Integrate(func, xmin, xmax, N[i])# segments

print2(A)

plt.figure()
plt.plot(N, A)
plt.plot(N, A, '*', color='Red')
plt.title("N vs. Integration Value")
plt.xlabel("N (number of segments)")
plt.ylabel("Integration Value")
plt.grid()
plt.xlim(0, 1100)
SAVE(1)
plt.show()

PDF("Assignment4.py", "Assignment4.pdf")