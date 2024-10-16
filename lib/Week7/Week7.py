from math import factorial

from lib.Header import *

# Monday Taylor Series Differentiation

def func(x):
    return np.e**x

def exTaylorSeries(x): # takes in function and x and return f(x+h)
    return 1 + x + x**2/factorial(2) + x**3/factorial(3) # + ...

v0 = 280        # ft/s : Muzzle Speed
g= 32.2       # ft/s^2 : Gravity Constant
# y0 = 0          # ft : starting height
# [x, y] = [0, 0] # ft : Target Coordinate

def Projectile(x, theta):
    return - .5 * g * ( x / (v0 * np.cos(theta * np.pi/180)) )**2 - x * np.tan(theta * np.pi/180)

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

'''
xlin = np.linspace(0, 1, 1000)

xder1 = forwardDiff(func, x)
xder2 = backwardDiff(func, x)
xder3 = centralDiff(func, x)

print(xder1)

plt.figure()
plt.plot(xlin[0:999], func(xlin[0:999]), label='func')
plt.plot(xlin[0:999], xder1, label='D1')
plt.plot(xlin[0:999], xder2, label='D2')
plt.plot(xlin[0:998], xder3, label='D3')
plt.legend()
plt.show()
'''