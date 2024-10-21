# Monday

import numpy as np
import matplotlib.pyplot as plt

g = 32.2 # ft/s/s
m = 180/g # slug
rho = 0.002377 # slug/ft^3
R = 10 # ft
A = np.pi * R**2 # ft^2
Cd = 1.75
CD = Cd * .5 * rho * A

h1 = 1e-5
time1 = np.arange(0, 40, h1)

x1 = np.zeros(len(time1))
x1[0] = 500

v1 = np.zeros(len(time1))
v1[0] = 60

def acc(v, T):
    return  g - ( CD * v**2 / m ) - T/m

for i in range(len(time1)-1):
    if (abs(x1[i]) < 50) & (v1[i] > 4):
        T = 170
    else:
        T = 0
    vhalf = v1[i] + .5 * h1 * acc(v1[i], T)
    v1[i+1] = v1[i] + h1 * acc(vhalf, T)
    x1[i+1] = x1[i] - h1 * v1[i]
    if (abs(x1[i])<=1):
        break

plt.figure()
plt.plot(time1, v1, label='v1')
plt.show()

plt.figure()
plt.plot(time1, x1, label='x1')
plt.show()

# Wednesday