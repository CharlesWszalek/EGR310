# Monday
from ftplib import ftpcp

# Wedneday - QUIZ

# Friday - Parachute

import numpy as np
import matplotlib.pyplot as plt

g = 32.2 # ft/s/s
m = 180/g # slug
rho = 0.002377 # slug/ft^3
R = 10 # ft
A = np.pi * R**2 # ft^2
Cd = 1.75 # drag coeff
CD = Cd * .5 * rho * A

h1 = .1
time1 = np.arange(0, 5, h1)
h2 = 1e-5
time2 = np.arange(0, 5, h2)

x1 = np.zeros(len(time1))
x2 = np.zeros(len(time2))
x1[0] = 0
x2[0] = 0

v1 = np.zeros(len(time1))
v2 = np.zeros(len(time2))
v1[0] = 60
v2[0] = 60

def acc(v):
    return g - ( CD * v**2 / m )

for i in range(len(time1)-1):
    v1[i+1] = v1[i] + h1 * acc(v1[i])
    x1[i+1] = x1[i] + h1 * v1[i]

for i in range(len(time2)-1):
    v2[i+1] = v2[i] + h2 * acc(v2[i])
    x2[i+1] = x2[i] + h2 * v2[i]

plt.figure()
plt.plot(time1, v1, label='v1')
plt.plot(time2, v2, label='v2')
plt.show()

plt.figure()
plt.plot(time1, x1, label='x1')
plt.plot(time2, x2, label='x2')
plt.show()