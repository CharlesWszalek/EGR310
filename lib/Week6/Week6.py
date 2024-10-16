import numpy as np
from lib.Header import *

# MONDAY (SICK / ABSENT)

def f(x):
    return np.sqrt(1-x**2)

def Guassian (func, x1, x2, N):
    x = np.linspace(x1, x2, N + 1)
    zta1 = 1 / np.sqrt(3)
    zta2 = -1 / np.sqrt(3)
    area = 0
    for i in range(len(x)-1):
        a = x[i]
        b = x[i+1]
        bma2 = (b-a)/2
        apb2 = (b+a)/2
        x1 = apb2 + bma2 * zta1
        x2 = apb2 + bma2 * zta2
        area += bma2 * (func(x1) + func(x2))
    return area

N = 1

gauss_area = Guassian(f, 0, 1, N)

exact = np.pi/4

print2(f"EXACT:          {exact}")
print2(f"Gaussian:       {gauss_area}")
print2(f"Percent Error:  {(gauss_area - exact)*100/exact:.10f} %\n")

PDF()

# WEDNESDAY (HES SICK, NO CLASS)

# FRIDAY EXAM 1