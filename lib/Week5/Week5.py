import numpy as np
import sympy as sym

# MONDAY
def f(x):
    return np.sqrt(1-x**2)


def Trap_Int(func, x1, x2, N):
    x = np.linspace(x1, x2, N + 1)
    y = func(x)
    area = 0
    for i in range(len(x)-1):
        area += (y[i] + y[i+1])/2 * (x2-x1)/N
    return area


# WEDNESDAY
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


N = 1000 # segments

trap_area   = Trap_Int(f, 0, 1, N)
simp13_area = Simp13_Int(f, 0, 1, N)
simp38_area = Simp38_Int(f, 0, 1, N)

exact = np.pi/4

print(f"EXACT:          {exact}\n")
print(f"Trapezoid:      {trap_area}")
print(f"Difference:     {exact - trap_area:.10f}\n")
print(f"Simpson's 1/3:  {simp13_area}")
print(f"Difference:     {exact - simp13_area:.10f}\n")
print(f"Simpson's 3/8:  {simp38_area}")
print(f"Difference:     {exact - simp38_area:.10f}\n")
