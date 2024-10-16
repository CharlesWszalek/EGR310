import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
from lib.Header import *

table = pd.read_excel("C81_CL_Table.xlsx", header = None, skiprows = 0)

aoa = table.iloc[1:, 0].values                          # first column
print("Mach:", aoa)
mach = table.iloc[0, 1:].values                         # first row
print("AoA:", mach)
dat = table.iloc[1:, 1:].values                         # everything else
print("Data Shape:", dat.shape)

points = [[i, j] for j in aoa for i in mach]            # creating point pairs that \
                                                        # go in order left to right then down
print("Points: ", points)

data = dat.ravel()
# for i in range(len(dat)):
#    data = np.concatenate((data, dat[i]))              # ravel works instead
# print("Data: ", data)

poi_mach = .45
poi_aoa = 10
poi = [poi_mach, poi_aoa]                               # point of interest

# CL_poi = RegularGridInterpolator((mach, aoa), dat, method='linear') # not a regular grid, won't work!
z_interp = griddata(points, data, poi, method='linear') # linear interpolation of the data
print("Z: ", z_interp)

X, Y = np.meshgrid(mach, aoa)
# print("X: ", X, "\nY: ", Y)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Mach')
ax.set_ylabel('Angle of Attack (AoA)')
ax.set_zlabel('Data Values')

ax.scatter(.45, 10, z_interp, c='r', marker='o',s=100)
ax.plot_surface(X, Y, dat, alpha=0.5, cmap='cool_r')
SAVE(1)

plt.show()