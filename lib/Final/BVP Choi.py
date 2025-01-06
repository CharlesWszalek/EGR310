import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp

# Function to calculate the rocket trajectory and find the range difference
def ivp_rocket(speed, target):
    te = 150
    ang = 60  # Launch angle in degrees
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
vini_0 = 100  # Initial guess for velocity
target = 50 * 5280  # Target distance in feet

# Solve for the initial velocity that hits the target
solution = root_scalar(lambda vini: ivp_rocket(vini, target)[0], x0=vini_0, x1=vini_0 + 10)
vini = solution.root

# Solve the trajectory with the determined velocity
_, t45, vx45 = ivp_rocket(vini, target)

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