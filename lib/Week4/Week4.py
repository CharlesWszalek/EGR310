import matplotlib.pyplot as plt
import numpy as np

# MONDAY

'''
1 -1000 0
2 -137 837
3 725 1349
4 1588 1853
5 2451 1969
6 3313 1998
7 4176 1820
8 5039 1468
9 5901 799
10 6764 0
'''

points = [[-1000, 0],
          [-137, 837],
          [725, 1349],
          [1588, 1853],
          [2451, 1969],
          [3313, 1998],
          [4176, 1820],
          [5039, 1468],
          [5901, 799],
          [6764, 0]]

[[x1, b1],
 [x2, b2],
 [x3, b3]] = [[725, 1349],
              [1588, 1853],
              [2451, 1969]]

xrange = np.linspace(0,8000,8000)


A = np.array([[1, x1, x1**2],
              [1, x2, x2**2],
              [1, x3, x3**2]])

b = np.array([b1,
              b2,
              b3])

a = np.matmul(np.linalg.inv(A), b)

print(a)

plt.figure()
plt.plot(xrange, a[0] + a[1]*xrange + a[2]*xrange**2)
plt.plot([coord[0] for coord in points], [coord[1] for coord in points], '.')
plt.show()

# WEDNESDAY

# FRIDAY

mach = np.readtxt("C81_CL_Table.xlsx")