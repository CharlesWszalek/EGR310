#------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
L = 22.5    # ft
P = 25000   # LB
h = 2       # ft
Mtip = 1000 # ft-lb
sca = 3500  # max strain in microstrain

# Calculations for EInode and EImin
EInode = P * 1.0e6 / sca
EImin = Mtip * 1.0e6 / sca

# Initial conditions for displacement and slope
sv0 = [0, 0]

# Define the function d(sv)/dx
def dsvdx(x, sv, P, L, EInode, EImin, Mtip):
    EI = EInode * (L - x) + EImin
    M = P * (L - x) + Mtip
    return [sv[1], M / EI]

# Solve the differential equation
sol = solve_ivp(dsvdx, [0, L], sv0, args=(P, L, EInode, EImin, Mtip), dense_output=True)
x = np.linspace(0, L, 100)
sv = sol.sol(x)

# Calculate Bending Stiffness EI, Bending Moment M, and Strain
EI = EInode * (L - x) + EImin
M = P * (L - x) + Mtip
strain = M * 0.5 * h / EI

# Plotting
plt.figure(1)
plt.plot(x, sv[1], 'b-')
plt.xlabel('Span (ft)')
plt.ylabel('Displacement (ft)')
plt.title('Wing Deflection')
plt.grid(True)
#plt.gca().set(fontsize=14, fontweight='bold', linewidth=2)
plt.gca().lines[-1].set_linewidth(2)

plt.figure(2)
plt.plot(x, strain, 'b-')
plt.xlabel('Span (ft)')
plt.ylabel('Strain (in/in)')
plt.title('Wing Strain Distribution')
plt.grid(True)
#plt.gca().set(fontsize=14, fontweight='bold', linewidth=2)
plt.gca().lines[-1].set_linewidth(2)

plt.figure(3)
plt.plot(x, EI, 'b-')
plt.xlabel('Span (ft)')
plt.ylabel('Bending Stiffness')
plt.title('Wing Bending Stiffness Distribution')
plt.grid(True)
#plt.gca().set(fontsize=14, fontweight='bold', linewidth=2)
plt.gca().lines[-1].set_linewidth(2)

plt.figure(4)
plt.plot(x, M, 'b-')
plt.xlabel('Span (ft)')
plt.ylabel('Bending Moment [lb-ft]')
plt.title('Wing Bending Moment Distribution')
plt.grid(True)
#plt.gca().set(fontsize=14, fontweight='bold', linewidth=2)
plt.gca().lines[-1].set_linewidth(2)

plt.show()