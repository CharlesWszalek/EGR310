# Interpolate the Lift Coefficient, CL, for a Mach number of 0.35 and an
# Angle of Attack (AoA) of 9.078 degrees using the C81 Lift Coefficient Data table provided on CANVAS.

import numpy as np
import pandas as pd
import openpyxl
import scipy
import matplotlib.pyplot as plt

MachNum = .35
AoA = 9.078
'''
Table = pd.read_excel("C81_CL_Table.xlsx")

X = [8,10]
Y = [.3,.4]
Z = [0.88, 0.907,
        1.1, 1.082]
'''

pt1 = [8, .8935]
pt2 = [10, 1.064]

y = np.linspace(pt1[1], pt2[1], 1000)
x = np.linspace(pt1[0], pt2[0], 1000)

plt.plot(x, y)
plt.xlim(9.070, 9.080)
plt.grid()
plt.show()

