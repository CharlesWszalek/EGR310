from lib.Header import *

# Monday

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

def acc1(v, T):
    return  g - ( CD * v**2 / m ) - T/m

for i in range(len(time1)-1):
    if (abs(x1[i]) < 50) & (v1[i] > 4):
        T = 170
    else:
        T = 0
    vhalf = v1[i] + .5 * h1 * acc1(v1[i], T)
    v1[i+1] = v1[i] + h1 * acc1(vhalf, T)
    x1[i+1] = x1[i] - h1 * v1[i]
    if (abs(x1[i])<=1):
        break

plt.figure()
plt.plot(time1, v1, label='v1')

plt.figure()
plt.plot(time1, x1, label='x1')

# Wednesday: Heun's Method (Compare this to Week8.py)
'''
h2 = 1e-4
time2 = np.arange(0, 5, h2)

x2 = np.zeros(len(time2))
x2[0] = 0

v2 = np.zeros(len(time2))
v2[0] = 60

def acc2(v):
    return  g - ( CD * v**2 / m )

for i in range(len(time1)-1):
    v1[i+1] = v1[i] + h2 * acc2(v1[i])
    x1[i+1] = x1[i] + h2 * v1[i]

for i in range(len(time2)-1):
    v_predictor = v2[i] + h2 * acc2(v2[i])
    v_corrector = v2[i] + .5 * h2 * ( acc2(v2[i]) + acc2(v_predictor) )
    for j in range(100): # while loop but want to avoid an infinite loop
        a = acc2(v_corrector)
        v_predictor = v2[i] + h2 * a
        a = .5 * (a + acc2(v_predictor))
        v_corrector = v2[i] + h2 * a
        if (abs(v_predictor - v_corrector) < 1.0e-6):
            v2[i+1] = v_corrector
            break

plt.figure()
plt.plot(time2, v2, label='v1')
'''
# ALSO MATRIX WORK
'''
g = 32.2 # ft/s/s
m = 180/g # slug
rho = 0.002377 # slug/ft^3
R = 10 # ft
A = np.pi * R**2 # ft^2
Cd = 1.75
CD = Cd * .5 * rho * A

h3 = 1e-4
time3 = np.arange(0, 5, h3)

vz = np.zeros((2, len(time3)))
vz[0][0] = 0
vz[0][1] = 60 # MIGHT BE 1 then 0 ???

# print(f"vz: {vz}")

def vzdot(vz):
    a = np.array([g - ( CD * vz[0]**2 / m ), vz[0]])
    # print(f"vzdot: {a}")
    return a

for i in range(len(time3)-1):
    vz_half = vz[:, i] + .5 * h3 * vzdot(vz[:, i])
    vz[:, i+1] = vz[:, i] - h3 * vzdot(vz_half)

plt.figure()
plt.plot(time3, vz[0], label='v1')
plt.show()
'''
# FRIDAY

te = 30
h4 = 1e-4
time4 = np.arange(0, te, h4)

vx1 = np.zeros((2, len(time4)))
vx1[0][0] = 0
vx1[1][0] = 1

vx2 = np.zeros((2, len(time4)))
vx2[0][0] = 0
vx2[1][0] = 1

vx3 = np.zeros((2, len(time4)))
vx3[0][0] = 0
vx3[1][0] = 1
print(vx3)

omega = 2 * np.pi / 180

def vx_dot (fom, vx):
    com = .2
    kom = 39
    vxdot = np.matmul([[-com, -kom], [1, 0]],vx) + [fom, 0]
    return vxdot

for i in range(len(time4)-1):
    fom1 = 0 # f over m
    fom2 = 10 * np.cos(.5 * omega * time4[i])
    fom3 = 10 * np.cos(omega * time4[i])
    vx_half1 = vx1[:, i] + .5 * h4 * vx_dot(fom1, vx1[:, i])
    vx1[:, i+1] = vx1[:, i] - h4 * vx_dot(fom1, vx1[:, i])
    vx_half2 = vx2[:, i] + .5 * h4 * vx_dot(fom2, vx2[:, i])
    vx2[:, i + 1] = vx2[:, i] - h4 * vx_dot(fom2, vx2[:, i])
    vx_half3 = vx3[:, i] + .5 * h4 * vx_dot(fom3, vx3[:, i])
    vx3[:, i + 1] = vx3[:, i] - h4 * vx_dot(fom3, vx3[:, i])

plt.figure()
plt.plot(time4, vx1[0], label='vx[0]')
plt.legend()

plt.figure()
plt.plot(time4, vx1[0], label='vx1[1]')
plt.legend()

plt.figure()
plt.plot(time4, vx1[1], label='vx1[0]')
plt.legend()

plt.figure()
plt.plot(time4, vx2[0], label='vx2[0]')
plt.legend()

plt.figure()
plt.plot(time4, vx2[1], label='vx2[1]')
plt.legend()

plt.figure()
plt.plot(time4, vx1[0], label='vx3[0]')
plt.legend()

plt.figure()
plt.plot(time4, vx1[0], label='vx3[1]')
plt.legend()

plt.show()
