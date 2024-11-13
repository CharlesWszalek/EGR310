import numpy as np

com = 0.2
kom = 39

A = [[-com, -kom],
     [1,      0]]

print(A)

time4 = np.linspace(0, 1, 10)
vx1 = np.zeros((2, len(time4)))

print(vx1)