"""
Charles Wszalek Week 2 Practice Code
"""

import numpy as np
import matplotlib.pyplot as plt

# Example 1 -- Projectile Motion -- HW 2

theta = 45 # degrees
v_0 = 1 # ft/s or m/s
g = 9.81 # m/s/s
g = 32.2 # ft/s/s

target = [300, 6]

Loc = [0, 0]

def AngleFinder(start, finish, v_0):
    angle = 1
    return angle

# Exmaple 2 -- Max Beam Difflection

def polySlope(a):
    return 5*a**4 - 9*a**2 + 4*a # Slope EQ

def polyYoc(a):
    return a**5 - 3*a**3 + 2*a**2 # Displacement EQ

def fn_bisection(xmin, xmax, mypoly, counter): # BISECTION METHOD
    tol = 1e-10
    counter += 1
    xf = 0
    yf = 0

    xa = (xmin + xmax)/2
    ya = mypoly(xa)

    if (abs(ya) < tol):
        xf = xa
        yf = ya
        return [xf, yf, counter]

    if (ya * mypoly(xmin) < 0):
        xmax = xa
    else:
        xmin = xa

    [xf, yf, counter] = fn_bisection(xmin, xmax, mypoly, counter)

    return [xf, yf, counter]

def fn_secant(x0, x1, mypoly, counter):  # SECANT METHOD
    tol = 1e-10
    counter += 1

    y0 = mypoly(x0)
    y1 = mypoly(x1)

    x2 = x1 - (y1 * (x1 - x0)) / (y1 - y0)
    y2 = mypoly(x2)

    if abs(y2) < tol:
        xf = x2
        yf = y2
        return [xf, yf, counter]

    [xf, yf, counter] = fn_secant(x1, x2, mypoly, counter)

    return [xf, yf, counter]

amin = .05
amax = .95
counter = 1

af = 0
dyaf = 0

[af, dyaf, counter] = fn_secant(amin, amax, polySlope, counter)

print([af, dyaf, counter])

a = np.linspace(amin, amax, 1000)
YoC = polyYoc(a)

plt.figure()
plt.plot(a, YoC, '-', linewidth=2)
plt.plot(af, polyYoc(af), 'o', markersize=10)
plt.title(r'Graphical Cross Check', fontsize=14, weight='bold')
plt.xlabel(r'$\alpha$ = x/L', fontsize=14, weight='bold')
plt.ylabel('Beam difflection, y/C', fontsize=14, weight='bold')
plt.grid()
plt.show()
plt.close()





