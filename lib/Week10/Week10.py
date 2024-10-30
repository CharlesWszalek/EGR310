from lib.Header import *

# MONDAY RUNGE KUTTA and ODE45
# parachute
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def vdot45(t, v):
    g = 32.2
    CDm = .1169
    return g - CDm * v**2

vini = [60]
tspan = [0, 5]
sol = solve_ivp(vdot45, tspan, vini)

plt.figure()
plt.plot(sol.t, sol.y[0])
plt.show()

# beam
def dsvdx(x, sv, P, L, EI):
    # return np.matmul([[0, 0], [1, 0]], sv[:,0]) + [P * (L-x)/EI, 0] # OLD
    # Define the system matrix
    A = np.array([[0, 0],
                  [1, 0]])
    print(np.shape(A))
    # Calculate the derivative
    print("sv: ", np.array(sv).reshape(-1, 1), np.shape(np.array(sv).reshape(-1, 1)))
    dsvdx1 = np.matmul(A, np.array(sv).reshape(-1, 1))
    print("a: ", dsvdx1)
    dsvdx1 = np.add(dsvdx1, np.array([[P * (L - x) / EI], [0]]))
    print("b: ", dsvdx1)
    return dsvdx1.flatten()
L = 10
P = 100
EI = 1e5
sv0 = np.array([0, 0])
t_span = (0, L)

sol2 = solve_ivp(dsvdx, t_span, sv0.flatten(), args=(P, L, EI))
plt.figure()
plt.plot(sol2.t, sol2.y[0])
plt.show()

dx = 1
tspan = [0, L, dx]

# WEDNESDAY Rocket Launcher


