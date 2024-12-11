
"""
----------------------
FINAL REVIEW AND COMPILATION
----------------------
"""

"""----------- IMPORTS -----------"""
import numpy as np

"""----------- ROOT FINDING -----------"""

# Week 1

def fn_cross_check(N, xb_old, mypoly, counter): # "Week 1"
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

# Week 2

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
