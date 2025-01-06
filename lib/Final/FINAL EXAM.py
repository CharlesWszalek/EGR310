import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp

# QUESTION 1
'''
def f1(x):
    return np.cos(x) * np.cosh(x) + 1

root1 = root_scalar(f1, x0=0, x1=3)
root2 = root_scalar(f1, x0=3, x1=5)

print(root1, root2)

xdata = np.arange(0, 5, .01)

plt.figure()
plt.plot(xdata, f1(xdata))
plt.grid()
plt.show()
'''

# QUESTION 2
'''
r = 2.0  # ft
L = 5.0  # ft
Vt = 50  # ft^3

def volume(h):
    res = Vt - L * (r**2 * np.acos((r-h)/(r)) - (r-h) * np.sqrt(2*r*h - h**2))
    return res

root1 = root_scalar(volume, x0=0, x1=2*r)
print(root1)

xdata = np.linspace(0, 2*r, 100)

plt.figure()
plt.plot(xdata, volume(xdata))
plt.grid()
plt.show()
'''

# QUESTION 3
'''
def Simp38_Int(func, x1, x2, N):
    x = np.linspace(x1, x2, N + 1)
    h = (x[1]-x[0])/3
    area = 0
    for i in range(len(x)-1):
        area += 3*h/8 * (func(x[i]) + 3*func(x[i]+h) + 3*func(x[i]+(2*h)) + func(x[i+1]))
    return area

def eq(x):
    return np.sin(x) * np.sin(2*x)

area = Simp38_Int(eq, 0, np.pi, 10000)
print(area)

xdata = np.linspace(0, np.pi, 1000)

plt.figure()
plt.plot(xdata, eq(xdata))
plt.grid()
plt.show()
'''

# QUESTION 4
L  = 120 # ft
def EI(x):
    return 5.08729e6 * (1-.07 * (x/L))**4
P  = 100 # lbs

n = 100
h = L/n
x = np.linspace(0, L, n+1)
A = np.zeros((n-1,n-1))
b = np.zeros((n-1,1))

A[0,0] = -2
A[0,1] = 1
A[n-2, n-3] = 1
A[n-2, n-2] = -2

b[0,0] = h
b[n-2,0] = h



for i in range(1,n-2): # is this right???
    A[i,i-1] = 1
    A[i,i] = -2
    A[i,i+1] = 1
    if x[i] < 48:
        b[i, 0] = x[i+1]
    else:
        b[i, 0] = L - x[i + 1]

EId = EI(x)

v = np.linalg.inv(A) @ b

for i in range(len(v)):
    v[i] = v[i] * (h ** 2 / EId[i]) * (P/2)

vfull = np.concatenate(([[0]], v, [[0]]))

plt.figure()
plt.plot(x, abs(vfull))
plt.xlabel("x inches")
plt.ylabel("vertical disp inches")
plt.title("SS Beam, FDM")
plt.grid()
# plt.ylim(.8, 1)
plt.show()

print(max(abs(vfull)))

# QUESTION 5
'''
# Function to calculate the rocket trajectory and find the range difference
def ivp_rocket(ang, target):
    te = 350
    # ang = 60  # Launch angle in degrees
    speed = 502.2
    vx0 = [
        speed * np.cos(np.radians(ang)),  # Initial horizontal velocity
        speed * np.sin(np.radians(ang)),  # Initial vertical velocity
        0,  # Initial x-position
        100  # Initial y-position
    ]
    tspan = [0, te]

    # Define the event function to stop integration when y < 0
    def halt_check(t, vx):
        return vx[3]  # Y-position
    halt_check.terminal = True
    halt_check.direction = -1

    # Solve the ODE
    result = solve_ivp(Rocket_ode, tspan, vx0, max_step=0.05, events=halt_check)

    t45, vx45 = result.t, result.y.T

    # Calculate the range difference
    range_val = vx45[-1, 2]  # Final x-position
    res = target - range_val

    return res, t45, vx45

# Rocket 1st Order ODE Equation
def Rocket_ode(t, vx):
    g = 32.2  # Acceleration due to gravity (ft/s^2)
    fe = 0.1  # Effective drag area (ft^2)
    cd = 0.1  # Drag coefficient
    m = 50 / 32.2  # Mass (slugs)
    rho = 0.002377  # Air density (slug/ft^3)

    X, Y = vx[0], vx[1]
    speed = np.sqrt(X**2 + Y**2)
    drag = 0.5 * rho * fe * cd * speed**2
    dragom = drag / m

    tm = 3 * 32.2 - dragom
    cosp = X / np.sqrt(X**2 + Y**2) if speed != 0 else 0
    sinp = Y / np.sqrt(X**2 + Y**2) if speed != 0 else 0

    dvxdt = [
        tm * cosp,  # Horizontal acceleration
        tm * sinp - g,  # Vertical acceleration
        X,  # X-position derivative
        Y   # Y-position derivative
    ]
    return dvxdt

# Initial guess and target distance
ang_0 = 30  # Initial guess for velocity
target = 50 * 5280  # Target distance in feet

# Solve for the initial velocity that hits the target
solution = root_scalar(lambda angle: ivp_rocket(ang_0, target)[0], x0=ang_0, x1=ang_0+46)
angle = solution.root

# Solve the trajectory with the determined velocity
_, t45, vx45 = ivp_rocket(53, target)

# Calculate velocity magnitude
d_velo = np.sqrt(vx45[:, 0]**2 + vx45[:, 1]**2)

# Plot results
# Rocket Speed
plt.figure(10)
plt.plot(t45, d_velo / 1125, 'r-', linewidth=2)
plt.xlabel('Time [sec]')
plt.ylabel('Rocket Speed [Mach Number]')
plt.title('Rocket Speed w/ solve_ivp')
plt.grid(True)
plt.gca().tick_params(labelsize=14, width=2)
plt.show()

# Trajectory
plt.figure(1)
plt.plot(vx45[:, 2] / 5280, vx45[:, 3] / 5280, 'r-', linewidth=2)
plt.xlabel('X (Mile)')
plt.ylabel('Y (Mile)')
plt.title('Trajectory w/ solve_ivp')
plt.grid(True)
plt.axis('equal')
# plt.xlim(40, 60)
plt.gca().tick_params(labelsize=14, width=2)
plt.show()

# X and Y Positions vs Time
plt.figure(2)
plt.plot(t45, vx45[:, 2], 'r-', label='x')
plt.plot(t45, vx45[:, 3], 'b-', label='y')
plt.xlabel('Time [sec]')
plt.ylabel('Position (ft)')
plt.title('Trajectory Components w/ solve_ivp')
plt.legend()
plt.grid(True)
plt.gca().tick_params(labelsize=14, width=2)
plt.show()
'''