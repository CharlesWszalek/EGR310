# MONDAY BVP: Parachute
'''
import numpy as np
import scipy.optimize as sci
from scipy.integrate import solve_ivp
from scipy.linalg import solve

m = 150/32.2 # mass
cd = 1.75
rho = .002377 # slug/ft^3
R = 5
CD = cd * .5 * rho * (np.pi * R**2)
vini_0 = 60

class prop:
    def __init__(self, g, CDm, tspan, zini, zfin):
        self.g = g
        self.CDm = CDm
        self.tspan = tspan
        self.zini = zini
        self.zfin = zfin

p = prop()
p.g = 32.2
p.CDm = CD/m
p.tspan = [0, 3]
p.zini = 100
p.zfin  = 0

def ivp(prop, vini):
    g = prop.g
    CDm = prop.CDm
    tspan = prop.tspan
    zini = prop.zini
    zfin = prop.zfin
    A45 = solve_ivp(parachute_ode, tspan, [vini, zini], args=(g, CDm))
    res = zfin - A45.y[-1][1]

def parachute_ode (t, vx, g, CDm):
    dvxdt = np.zeros(2)
    dvxdt[0] = g - CDm * vx[0]**2
    dvxdt[1] = -vx[0]
    return dvxdt

vini = sci.newton(ivp(prop, vini), vini_0)
'''
# WEDNESDAY Rocket Launcher

import numpy as np
import scipy.optimize
from scipy.integrate import solve_ivp
import scipy.optimize as sci

# Constants
def dvxdt(t, vx):
    g = 32.2  # gravitational acceleration (ft/s^2)
    fe = 0.1  # frontal area (ft^2)
    cd = 0.1  # drag coefficient
    m = 50 / 32.2  # mass (slug)
    rho = 0.002377  # air density (slug/ft^3)
    X = vx[0]
    Y = vx[1]
    speed  = np.sqrt(X**2 + Y**2)
    drag = .5 * rho * fe * cd * speed**2
    dragom = drag/m
    tm = 3 * 32.2 - dragom
    #tm = 0
    cosp =
    sinp =

    dvxdt =
    return dvxdt

def halt_condition(t, y):
    return y[0] < 0.01

def ivp_rocket(speed, target):
    te = 150
    ang = 60
    vx0 = [speed * np.cos(ang*np.pi/180), speed * np.sin(ang*np.pi/180), 0,  100]
    tspan = [0,te]
    [t45, vx45] = solve_ivp(rocket_ode, tspan, vx0, args=(speed, target), events=halt_condition)

vini_0 = 100
target = 50 * 5280  # target distance in feet (50 miles)
vini = sci.newton(ivp_rocket)

# Rocket ODE function
def rocket_ode(t, vx):
    X, Y, Vx, Vy = vx
    speed = np.sqrt(Vx ** 2 + Vy ** 2)
    drag = 0.5 * rho * fe * cd * speed ** 2
    dragom = drag / m

    dXdt = Vx
    dYdt = Vy
    dVxdt = - (dragom * Vx / speed)  # Drag in x direction
    dVydt = -g - (dragom * Vy / speed)  # Gravity + Drag in y direction

    return [dXdt, dYdt, dVxdt, dVydt]


# Function to compute the range error for a given initial speed
def ivp_rocket(vini, target):
    te = 150  # end time for integration
    ang = 60  # launch angle in degrees
    Vx0 = vini * np.cos(np.radians(ang))
    Vy0 = vini * np.sin(np.radians(ang))
    vx0 = [0, 0, Vx0, Vy0]  # initial conditions [X, Y, Vx, Vy]

    # Solve the ODE
    sol = solve_ivp(rocket_ode, [0, te], vx0, events=halt_check, rtol=1e-5)

    # Check if we reached the ground (Y=0) and get the final range
    if sol.status == 1:  # Halted due to event (Y < 0)
        final_x = sol.y[0, -1]
        return target - final_x
    else:
        return target  # Did not reach the ground, so return a large error


# Use root_scalar to find the initial speed that brings us to the target distance
vini_0 = 100  # Initial guess for root finding
result = root_scalar(ivp_rocket, args=(target,), bracket=[vini_0 / 2, vini_0 * 2])

# Display the result
if result.converged:
    print(f"Initial Rocket Speed: {result.root:.2f} ft/s")
else:
    print("Failed to find an initial speed that meets the target.")
'''

import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# Given parameters
m = 150 / 32.2  # mass
cd = 1.75
rho = 0.002377  # slug/ft^3
R = 5
CD = cd * 0.5 * rho * (np.pi * R**2)
CDm = CD / m
vini_0 = 60

# Properties of the problem
class prop:
    def __init__(self, g, CDm, tspan, zini, zfin):
        self.g = g
        self.CDm = CDm
        self.tspan = tspan
        self.zini = zini
        self.zfin = zfin

p = prop(g=32.2, CDm=CDm, tspan=[0, 3], zini=100, zfin=0)

# Define the ODE system for solve_bvp
def parachute_ode(t, vx):
    dvxdt = np.zeros((2, t.size))
    dvxdt[0] = p.g - p.CDm * vx[0]**2  # velocity derivative
    dvxdt[1] = -vx[0]                  # position derivative
    return dvxdt

# Boundary conditions for the position and velocity
def bc(vx_ini, vx_fin):
    return np.array([vx_ini[1] - p.zini,  # initial position
                     vx_fin[1] - p.zfin])  # final position

# Initial guess for the solution (constant initial guesses)
t_guess = np.linspace(p.tspan[0], p.tspan[1], 10)
vx_guess = np.zeros((2, t_guess.size))
vx_guess[0] = vini_0  # Initial guess for velocity

# Solve the boundary value problem
solution = solve_bvp(parachute_ode, bc, t_guess, vx_guess)

# Check if the solution was successful and print results
if solution.success:
    # Plot the velocity solution vx(t)
    plt.plot(solution.x, solution.y[0], label="vx(t)")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (ft/s)")
    plt.title("Parachute Falling with Air Resistance")
    plt.legend()
    plt.show()

    # Plot the position solution
    plt.plot(solution.x, solution.y[1], label="Position(t)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (ft)")
    plt.title("Position vs Time for Parachute Fall")
    plt.legend()
    plt.show()
else:
    print("The solver did not converge.")
'''
