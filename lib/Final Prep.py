
"""
----------------------
FINAL REVIEW AND COMPILATION
----------------------
"""

"""----------- IMPORTS -----------"""
import numpy as np
from scipy import optimize as sci

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

    # Newton-Raphson Method ??? uh oh
        # scipy.optimize.newton(func, x0, fprime=None, args=(), tol=1.48e-08, maxiter=50, fprime2=None, x1=None, rtol=0.0, full_output=False, disp=True)
        # where -> func(x, a, b, c, ...)

"""----------- OPTIMIZATION -----------"""

# Week 3

def golden_section_search(f, a, b, tol):
    R = (-1 + np.sqrt(5)) / 2
    h = b - a
    c = b - h*R
    d = a + h*R
    fc = f(c)
    fd = f(d)

    for i in range(10000):
        if (b - a) < tol:
            break
        if fc > fd:
            a = c
            c = d
            d = a + (b-a)*R
            fc = fd
            fd = f(d)
        else:
            b = d
            d = c
            c = b - (b-a)*R
            fd = fc
            fc = f(c)

    x_min = (a + b)/2
    f_min = f(x_min)

    return [x_min, f_min]

    # scipy.optimize.fmin(func, x0, args=(), xtol=0.0001, ftol=0.0001, maxiter=None, maxfun=None, full_output=0, disp=1, retall=0, callback=None, initial_simplex=None)

"""----------- CURVE FITTING -----------"""

# Week 3

def myLinReg(x, y):
    #-------------------------------------------------------
    # My Least Square Linear Regression Coding
    # y = bx + a
    #----------------------------------------------------
    C1 = len(x)
    C2 = 0
    C3 = 0
    C4 = 0
    f1 = 0
    f2 = 0
    # compute summations for coefficients
    for i in range(len(x)):
        C2 = C2 + x[i]
        C4 = C4 + x[i] ** 2
        f1 = f1 + y[i]
        f2 = f2 + x[i] * y[i]
    C3 = C2
    CMAT = [[C1, C2],
            [C3, C4]]
    FVEC = [f1,
            f2]
    AVEC = np.matmul(np.linalg.inv(CMAT), FVEC) # y = AVEC(2) * x + AVEC(1)
    a = AVEC[0]
    b = AVEC[1]
    return [b, a]

# Radar Tracking

[[x1, b1],
 [x2, b2],
 [x3, b3]] = [[725, 1349],
              [1588, 1853],
              [2451, 1969]]

def L1(x):
    return ((x-x2) * (x-x3)) / ((x1-x2) * (x1-x3))
def L2(x):
    return ((x-x1) * (x-x3)) / ((x2-x1) * (x2-x3))
def L3(x):
    return ((x-x1) * (x-x2)) / ((x3-x1) * (x3-x2))
def ylag(x): # Lagrange polynomial
    return L1(x) * b1 + L2(x) * b2 + L3(x) * b3

"""----------- INTERPOLATION -----------"""

# Interpolation

    # bisplrep(x, y, z, w=None, xb=None, xe=None, yb=None, ye=None, kx=3, ky=3, task=0, s=None, eps=1e-16, tx=None, ty=None, full_output=0, nxest=None, nyest=None, quiet=1)[source]

    # class RectBivariateSpline(x, y, z, bbox=[None, None, None, None], kx=3, ky=3, s=0)

"""----------- INTEGRATION -----------"""

# Week 5

def Trap_Int(func, x1, x2, N):
    x = np.linspace(x1, x2, N + 1)
    y = func(x)
    area = 0
    for i in range(len(x)-1):
        area += (y[i] + y[i+1])/2 * (x2-x1)/N
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

# Week 6

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

"""----------- DIFFERENTIATION -----------"""

# Week 7

def TaylorSeries1st(func, x, h): # takes in function and x and return f(x+h)
    return (func(x)
            + h * func(x) # need 1st derivative
            + h**2/factorial(2) * func(x) # need 2nd derivative
            + x**3/factorial(3) * func(x)) # need 3rd derivative

def forwardDiff(func, x):
    h = (max(x) - min(x)) / len(x)
    derivative = []
    for i in range(len(x)-1):
        derivative.append((func(x[i+1]) - func(x[i])) / h)
    # derivative = derivative.append(1)
    return derivative

def backwardDiff(func, x):
    h = (max(x) - min(x)) / len(x)
    derivative = []
    for i in np.arange(len(x)-1)+1:
        derivative.append((func(x[i]) - func(x[i - 1])) / h)
    return derivative

def centralDiff(func, x):
    h = (max(x) - min(x)) / len(x)
    derivative = []
    for i in np.arange(len(x)-2)+1:
        derivative.append((func(x[i + 1]) - func(x[i - 1])) / (2* h))
    return derivative

def forwardDiff1(func, x, h):
    return (func(x+h) - func(x)) / h

def backwardDiff1(func, x, h):
    return (func(x) - func(x-h)) / h

def centralDiff1(func, x, h):
    return (func(x+h) - func(x-h)) / (2*h)

"""----------- INITIAL VALUE PROBLEMS -----------"""

# Week 8

    # for i in range(len(time1)-1):
        # v1[i+1] = v1[i] + h1 * acc(v1[i])
        # x1[i+1] = x1[i] + h1 * v1[i]

# Week 9

    # Midpoint
for i in range(len(time1)-1):
    if (abs(x1[i]) < 50) & (v1[i] > 4):
        T = 170
    else:
        T = 0
    vhalf = v1[i] + .5 * h1 * acc1(v1[i], T)
    v1[i+1] = v1[i] + h1 * acc1(vhalf, T)
    x1[i+1] = x1[i] - h1 * v1[i]
    if (abs(x1[i])<=1):
        break

    # Heun's
for i in range(len(time1)-1):
    v1[i+1] = v1[i] + h2 * acc2(v1[i])
    x1[i+1] = x1[i] + h2 * v1[i]

for i in range(len(time2)-1):
    v_predictor = v2[i] + h2 * acc2(v2[i])
    v_corrector = v2[i] + .5 * h2 * ( acc2(v2[i]) + acc2(v_predictor) )
    for j in range(100): # while loop but want to avoid an infinite loop
        a = acc2(v_corrector)
        v_predictor = v2[i] + h2 * a
        a = .5 * (a + acc2(v_predictor))
        v_corrector = v2[i] + h2 * a
        if (abs(v_predictor - v_corrector) < 1.0e-6):
            v2[i+1] = v_corrector
            break

def vx_dot (fom, vx):
    com = 0.2
    kom = 39
    vxdot = np.matmul([[-com, -kom], [1, 0]],vx) + [fom, 0]
    return vxdot

    # Matrix Midpoint
for i in range(len(time4)-1):
    fom1 = 0 # f over m
    vx_half1 = vx1[:, i] + .5 * h4 * vx_dot(fom1, vx1[:, i])
    vx1[:, i + 1] = vx1[:, i] + h4 * vx_dot(fom1, vx1[:, i])

# Choi

def dsvdx(x, sv, P, L, EInode, EImin, Mtip):
    EI = EInode * (L - x) + EImin
    M = P * (L - x) + Mtip
    return [sv[1], M / EI]

"""----------- BOUNDARY VALUE PROBLEMS -----------"""

    # See MATLAB

