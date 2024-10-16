
from lib.Header import *

# QUESTION 1

v0 = 280 # ft/s : Muzzle Speed
g= 32.2 # ft/s^2 : Gravity Constant
y0 = 0 # starting height
[x, y] = [1900, 200] # ft : Target Coordinate


def Projectile(theta):
    return y + .5 * g * ( x / (v0 * np.cos(theta * np.pi/180)) )**2 - x * np.tan(theta * np.pi/180)


def fn_bisection(xmin, xmax, mypoly, counter): # BISECTION METHOD
    tol = 1e-10
    counter += 1
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

def fn_secant(x0, x1, mypoly, counter):  # UNUSED ONLY FOR TEST PURPOSES
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

maxdist = Projectile(-45)
print("The Mortar's maximum range is ", round(maxdist, 3), "ft. Which is ", round(maxdist * 0.3048, 3), "meters.")

anglemin = -50
anglemax = 70

N = 1000
counter = 0
angles1 = np.linspace(anglemin, anglemax, N)

[xf0, yf0, counter0] = fn_bisection(anglemin, anglemax, Projectile, counter)
print("That angle btween 50 and 70 degrees that hits the target is:", round(xf0,3), "degrees (orange point).")

plt.figure()
plt.grid()
plt.xlabel("Angle (degrees)")
plt.ylabel("Residue (ft)")
plt.title('Projectile', fontsize=14)
plt.plot(angles1, Projectile(angles1), linewidth=2)
plt.plot(xf0,yf0,'o', linewidth=20)
# plt.show()

# QUESTION 2

rho = 0.002308  # slug/ft^3, air density at 10,000 ft ALT.
A   = 174       # Wing Wetted Area , ft^2, Cessna 172
W   = 2450      # Gross WT, LB, Cessna 172
AR  = 6.5       # Wing Aspect Ratio
Cdo = 0.018     # Cd_node, Drag Coeff for zero lift
tol = 1e-6      # Tolerance of the minimization

def D(x):
    return -(1.6 * x **3 + 3 * x **2 - 2 * x) # I have a min fn but want max so i used -( ... )

def GoldenSearch(f, a, b, tol):
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


velocity = np.linspace(-2, 1, 1000)
[vmin, fmin] = GoldenSearch(D, -2, 1, tol)
print(vmin, fmin)

plt.figure()
plt.plot(velocity, D(velocity), label='drag as a function of velocity')
plt.plot(vmin, fmin, 'o', label='minimum')
plt.legend()
# plt.xlim(0)
# plt.ylim(0)
plt.title("Drag on a Cesna 172")
plt.xlabel("Velocity (ft/s)")
plt.ylabel("Drag")
plt.grid()
# plt.show()

# QUESTIONS 3 AND 4

p = 1.00000001

def f(x):
    return (p**2 + p) * x**4 + p*x + 1

def f2(px):
    return (2/9) * px **2 + (2/9) *px + 2


a = f(1/np.sqrt(3))
print(a)

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


velocity = np.linspace(-2, 2, 1000)
[vmin, fmin] = GoldenSearch(f2, -2, 2, tol)
print(vmin, fmin)

plt.figure()
plt.plot(velocity, f2(velocity), label='drag as a function of velocity')
plt.plot(vmin, fmin, 'o', label='minimum')
plt.legend()
# plt.xlim(0)
# plt.ylim(0)
plt.title("Drag on a Cesna 172")
plt.xlabel("Velocity (ft/s)")
plt.ylabel("Drag")
plt.grid()

gauss_area = Guassian(f2, -2, 2, N)

# exact = np.pi/4

# print(f"EXACT:          {exact}")
print(f"Gaussian:       {gauss_area}")
# print(f"Percent Error:  {(gauss_area - exact)*100/exact:.10f} %\n")

plt.show()