import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.integrate import solve_ivp

rho = 0.002377 # slug/ft3 Air Density
A   = 0.1 # ft2 Area
Cd  = 0.1 # Drag Coefficient
g   = 32.2
m   = 50 / g # lb/ ft/s2 Mass of the Rocket
T   = 3*m*g # =3*1.553*32.2 = 150 LB
xf  = 50*5280 # mi * ft/mi Target Distance
x0  = 0
y0  = 100 # ft Initial Height (Y initial) of the Rocket
theta = 60 * np.pi /180 # degree * rad/degree Rocket Shooting Angle
vini0 = 100

'''
Q1) (10 points) Determine the initial velocity required for a rocket to strike a target at 50miles.
- Solve BVP
- Report the calculated initial velocity value that satisfies the boundary condition.
'''

# Initial speed (can be defined based on the problem)
v0 = 300  # ft/s (assumed initial speed)
v0x = v0 * np.cos(theta)
v0y = v0 * np.sin(theta)

# Function defining the ODE system
def fun(t, y):
    # y[0] = x, y[1] = y, y[2] = v_x, y[3] = v_y

    dxdt = y[2]
    dydt = y[3]

    v = np.sqrt(y[2] ** 2 + y[3] ** 2)  # Velocity magnitude

    D = 0.5 * rho * Cd * A * v**2

    dvxdt = (T-D)/m * y[2]/v
    dvydt = (T-D)/m * y[3]/v - g

    return np.array([dxdt, dydt, dvxdt, dvydt])

# Boundary conditions
def bc(ya, yb):
    # ya corresponds to initial conditions and yb to final conditions
    return np.array([ya[0] - x0, ya[1] - y0, yb[0] - xf, yb[1]])

# Initial guess for the solution (initial conditions and a linear guess for the trajectory)
t = np.linspace(0, 100, 100)  # Time grid, adjust as needed
y_guess = np.zeros((4, t.size))
y_guess[0, :] = np.linspace(x0, xf, t.size)
y_guess[1, :] = np.linspace(y0, 0, t.size)  # y = 0 when it hits the ground
y_guess[2, :] = v0x
y_guess[3, :] = v0y

# Solve the BVP
sol = solve_bvp(fun, bc, t, y_guess)
sol2 = solve_ivp(fun, [0, 100], [0, 100, 433.26557, 493.69713], dense_output=True)

print(f"Initial Velocity: {np.sqrt(sol.sol(t)[2][0]**2 + sol.sol(t)[3][0]**2)}")
print(f"Initial Velocity X: {sol.sol(t)[2][0]}")
print(f"Initial Velocity Y: {sol.sol(t)[3][0]}")

print(f"Initial Velocity 2: {np.sqrt(sol2.y[2][0]**2 + sol2.y[3][0]**2)}")
print(f"Initial Velocity 2 X: {sol2.y[2][0]}")
print(f"Initial Velocity 2 Y: {sol2.y[3][0]}")

'''
Q2) (10 points) Create a plot showing the trajectory of the rocket that hits the target:
a) Plot X distance vs. Y distance (Report the plot)
b) Ensure that the Y distance is zero when the X distance reaches 50miles, or 50*5280 ft
'''

plt.figure()
plt.plot(sol.sol(t)[0], sol.sol(t)[1])
plt.title('Rocket Trajectory')
plt.xlabel('Distance (ft)')
plt.ylabel('Height (ft)')
plt.grid()
plt.show()

plt.figure()
plt.plot(sol2.y[0], sol2.y[1])
plt.title('Rocket Trajectory')
plt.xlabel('Distance (ft)')
plt.ylabel('Height (ft)')
plt.grid()
plt.show()

