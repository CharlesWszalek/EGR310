import numpy as np
import matplotlib.pyplot as plt
'''
# Monday - Optimization and Golden Section Search

def poly(x):
    return x**4 - 12*x**2 + 36

def function1(f, a, b, tol):
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

a = 0
b = 3
tol = 1e-6

x = np.arange(a, .001, b)

plt.figure()
plt.plot(x, poly(x))

[xmin, fmin] = function1(poly, a, b, tol)
print(xmin, fmin)
'''
# Wednesday - QUIZ

# Friday - Curve Fitting

x = np.random.random(100)*10

noise = np.random.randn(100)

y = 2*x + 3 + noise

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

[b, a] = myLinReg(x,y)
RY = b*x+a

plt.figure()
plt.plot(x, y, '.')
plt.plot(x, RY)
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sample Test Data')
plt.show()