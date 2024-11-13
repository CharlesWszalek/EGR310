# MONDAY: FDM Beam     ->  ->  ->      MAKE SURE YOU CAN DO A BEAM WITH A CHANGING EI
import numpy as np
import matplotlib.pyplot as plt

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

for i in range(1,n-2):
    A[i,i-1] = 1
    A[i,i] = -2
    A[i,i+1] = 1
    if x[i] < L/2:
        b[i, 0] = x[i+1]
    else:
        b[i, 0] = L - x[i + 1]
#print(A)
#print(b)

v = np.linalg.inv(A) @ b * (h**2/EI) * (P/2)

#print(v)

vfull = np.concatenate(([[0]], v, [[0]]))

#print(vfull)

plt.figure()
plt.plot(x, vfull)
plt.xlabel("x inches")
plt.ylabel("vertical disp inches")
plt.title("SS Beam, FDM")
plt.show()


