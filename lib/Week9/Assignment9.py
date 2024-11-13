from lib.Header import *
from scipy.integrate import solve_ivp

W   = 50000  # lbs
P   = W/2    # lbs
L   = 45/2   # ft
M0  = 1000   # lb ft
h   = 2      # ft
EIn = 7.14e6 # LB-ft^2
EIt = 2.86e5 # LB-ft^2

dist = np.linspace(0, L, 100)

M = P * (L-dist) + M0
EI =  EIn * (L-dist) + EIt
emax =  M * h * .5 / EI

plt.figure()
plt.title("Maximum strain along wing")
plt.plot(dist, emax)
plt.xlabel("Distance")
plt.ylabel("Maximum Strain")
SAVE(1)
plt.show()

def dsvdx(x, sv, P, L, EIn, EIt, M0):
    EI = EIn * (L - x) + EIt
    M = P * (L - x) + M0
    return [sv[1], M / EI]

sv0 = [0, 0]
density = np.linspace(0, L, 100)
sol = solve_ivp(dsvdx, [0, L], sv0, args=(P, L, EIn, EIt, M0), t_eval=density)

plt.figure()
plt.title("Difflection of wing")
plt.plot(sol.t, sol.y[0])
plt.xlabel("Distance")
plt.ylabel("Difflection")
SAVE(2)
plt.show()

PDF()
