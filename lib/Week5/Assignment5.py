import numpy as np
import matplotlib.pyplot as plt
from lib.Header import *

def Num(x):
    return x * np.sqrt(1 - x**2)
def Den(x):
    return np.sqrt(1 - x**2)

def Z(int_func, x1, x2, N):
    return int_func(Num, x1, x2, N) / int_func(Den, x1, x2, N)

def Trap_Int(func, x1, x2, N):
    x = np.linspace(x1, x2, N + 1)
    area = 0
    for i in range(len(x)-1):
        area += (func(x[i]) + func(x[i+1]))/2 * (x2-x1)/N
    return area

def Simp13_Int(func, x1, x2, N):
    x = np.linspace(x1, x2, N + 1)
    h = (x[1]-x[0])/2
    area = 0
    for i in range(len(x)-1):
        area += h/3 * (func(x[i]) + 4*func(x[i]+h) + func(x[i+1]))
    return area

def Simp38_Int(func, x1, x2, N):
    x = np.linspace(x1, x2, N + 1)
    h = (x[1]-x[0])/3
    area = 0
    for i in range(len(x)-1):
        area += 3*h/8 * (func(x[i]) + 3*func(x[i]+h) + 3*func(x[i]+(2*h)) + func(x[i+1]))
    return area

x1 = 0
x2 = 1
N = 10

Trap   = Z(Trap_Int,   x1, x2, N)
Simp13 = Z(Simp13_Int, x1, x2, N)
Simp38 = Z(Simp38_Int, x1, x2, N)

print2(f"Trapezoidal Integral: {Trap}")
print2(f"Simpson 1/3 Integral: {Simp13}")
print2(f"Simpson 3/8 Integral: {Simp38}")

PDF()