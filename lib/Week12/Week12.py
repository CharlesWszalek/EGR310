# MONDAY: FDM Simply Supported (SS) Beam   ->  ->  ->    MAKE SURE YOU CAN DO A BEAM WITH A CHANGING EI
import numpy as np
import matplotlib.pyplot as plt
from lib.Header import *

L  = 120 # in
EI = 5e6 # lb-in^2
P  = 100 # lbs

n = 10
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
    if x[i] < L/2:
        b[i, 0] = x[i+1]
    else:
        b[i, 0] = L - x[i + 1]

v = np.linalg.inv(A) @ b * (h**2/EI) * (P/2)

vfull = np.concatenate(([[0]], v, [[0]]))

plt.figure()
plt.plot(x, vfull)
plt.xlabel("x inches")
plt.ylabel("vertical disp inches")
plt.title("SS Beam, FDM")
SAVE(1)
plt.show()

# FRIDAY: Clamped Beam

L = 120 # in
P = 100 # lb
EI = 5e6 # lb in^2

exact = P*L**3 / (3*EI)

n = 10
h = L/n
x = np.linspace(0, L, n)
dx = x[1] - x[0]
A = np.zeros((n-1,n-1))
b = np.zeros((n-1,1))
A[0,0] = 2
A[1,0] = -2
A[1,1] = 1

sca = P * dx**2 / EI

b[0,0] = sca * (L - x[0])
b[1,0] = sca * (L - x[1])

for i in np.arange(2, n-1, 1):
    A[i, i-2] = 1
    A[i, i-1] = -2
    A[i, i] = 1
    b[i, 0] = (L-x[i])*sca

v = np.linalg.inv(A) @ b

vfull = np.concatenate(([[0]], v))

plt.figure()
plt.plot(x, vfull)
plt.xlabel("x inches")
plt.ylabel("vertical disp inches")
plt.title("Clamped Beam, FDM")
SAVE(2)
plt.show()

PDF()
