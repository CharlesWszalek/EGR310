import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from scipy.integrate import solve_ivp
from functools import partial

vini0 = 656.8530
xf = 50 * 5280

def ivp_rocket(speed):
    te = 150
    ang = 60
    vx0 = [speed * np.cos(ang*np.pi/180), speed * np.sin(ang*np.pi/180), 0,  100]
    tspan = [0,te]
    [t45, vx45] = solve_ivp(rocket_ode, tspan, vx0, args=(speed), events=halt_check)

    range = vx45[-1,2]
    return [t45, vx45]

def halt_check(t, vx):
    if vx[3] < 0:
        value = 0
        isterminal = 1
        direction = 0
    else:
        value = 1
        isterminal = 0
        direction = 0
    return [value, isterminal, direction]

def rocket_ode(t, vx):
    g = 32.2
    fe = .1
    cd = .1
    m = 50/g
    rho = .002377
    X = vx[0]
    Y = vx[1]
    speed = np.sqrt(X**2 + Y**2)
    drag = .5 * rho * fe * cd * speed**2
    dragom = drag/m
    tm = 3 * 32.2 - dragom
    cosp = X/speed
    sinp = Y/speed
    return [tm * cosp, tm * sinp - g, X, Y]

# ivp_rocket_partial = partial(ivp_rocket, target=xf)
vini = newton(lambda speed: ivp_rocket(speed)[0], vini0)
[t45, vx45] = ivp_rocket(vini)
velo = np.sqrt(vx45[:,0]**2 + vx45[:,1]**2)
