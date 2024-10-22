from lib.Header import *

def centralDiff(func, x):
    h = (max(x) - min(x)) / len(x)
    h = .1
    print(h)
    derivative = []
    for i in np.arange(len(x)-2)+1:
        derivative.append((func(x[i + 1]) - func(x[i - 1])) / (2* h))
    return derivative

a = 1500 # ft
time = [.9, 1, 1.1] # sec
alpha = np.array([54.8, 54.06, 53.34]) * np.pi/180 # rad
beta = np.array([65.59, 64.59, 63.62]) * np.pi/180 # rad

def x(index):
    return a * (np.tan(beta[index])) / (np.tan(beta[index]) - np.tan(alpha[index]))
def y(index):
    return a * (np.tan(beta[index])*np.tan(alpha[index])) / (np.tan(beta[index]) - np.tan(alpha[index]))

vx = centralDiff(x, range(len(time)))
vy = centralDiff(y, range(len(time)))

print(vx)
print(vy)

v = np.sqrt(vx[0]**2 + vy[0]**2)

print2(f"Velocity: {v}")

PDF()