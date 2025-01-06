import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# Constants
L = 10.0  # Length of the beam (in inches)
x_p = 5.0  # Position of the applied load (in inches)
P = 100.0  # Point load (in pounds)
EI = 5.0e6  # Flexural rigidity (in lb-in^2)

# Define the differential equations
def beam_ode(x, y):
    """
    Differential equations describing the beam deflection problem:
    y[0] = y (deflection)
    y[1] = y' (slope)
    y[2] = y'' (moment/EI)
    y[3] = y''' (shear/EI)
    """
    dydx = np.zeros_like(y)
    dydx[0] = y[1]  # Slope
    dydx[1] = y[2]  # Moment/EI
    dydx[2] = y[3]  # Shear/EI
    dydx[3] = np.where(np.isclose(x, x_p, atol=1e-6), -P / EI, 0.0)  # Point load modeled as a Dirac delta approximation
    return dydx

# Boundary conditions
def beam_bc(ya, yb):
    """
    Boundary conditions for the beam:
    - At x = 0: y(0) = 0 (no deflection), y'(0) = 0 (no slope at fixed support)
    - At x = L: y''(L) = 0 (no moment at roller support), y'''(L) = 0 (no shear at roller support)
    """
    return np.array([ya[0], ya[1], yb[2], yb[3]])

# Initial guess for the solution
x = np.linspace(0, L, 100)  # Discretize the beam length into 100 points
y_guess = np.zeros((4, x.size))  # Initial guess: zero deflection, slope, moment, and shear

# Solve the BVP using scipy.integrate.solve_bvp
solution = solve_bvp(beam_ode, beam_bc, x, y_guess)

# Check if the solution was successful
if solution.success:
    print("Solution converged!")
else:
    print("Solution did not converge.")

# Plot the results
x_plot = np.linspace(0, L, 500)  # Fine grid for smooth plotting
y_plot = solution.sol(x_plot)[0]  # Extract deflection (y) from the solution

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, label="Deflection (y)")
plt.axvline(x_p, color="red", linestyle="--", label="Point Load at x = {:.1f}".format(x_p))
plt.title("Beam Deflection under Point Load")
plt.xlabel("x (inches)")
plt.ylabel("Deflection y (inches)")
plt.legend()
plt.grid()
plt.show()
