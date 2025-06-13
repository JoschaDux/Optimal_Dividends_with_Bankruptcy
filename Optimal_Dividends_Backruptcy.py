#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 22:46:44 2025

@author: joschaduchscherer
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Define model parameters
class Model:
    def __init__(self):
        self.mu = 2 #Drift
        self.gam = 0.1 #Absolute Risk-Aversion
        self.delta = 0.05 #Time-Preference Rate
        self.lam = 0.1 #Jump-intensity
        self.b = 5 #Loss Size

# Define grid parameters
class Grid:
    def __init__(self):
        self.xmin = 0
        self.xmax = 10
        self.tmin = 0
        self.tmax = 100
        self.Nx = 100
        self.Nt = 100000
        self.dt = (self.tmax - self.tmin) / self.Nt
        self.dx = (self.xmax - self.xmin) / self.Nx
        self.nx = np.arange(self.Nx + 1)
        self.nt = np.arange(self.Nt + 1)
        self.x = self.xmin + self.dx * self.nx

model = Model()
grid = Grid()

# Define policy function
def policy(V_ns):
    V_ns_x = (np.roll(V_ns, -1) - np.roll(V_ns, 1)) / (2 * grid.dx)
    V_ns_x[0] = 1
    
    # Linear extrapolation at upper bound
    slope = (V_ns_x[-2] - V_ns_x[-3]) / (grid.x[-2] - grid.x[-3])
    intercept = V_ns_x[-2] - slope * grid.x[-2]
    V_ns_x[-1] = slope*grid.x[-1]+intercept
    d = -np.log(V_ns_x) / model.gam
    d[d <= 0] = 0
    return d

# Define function for the coefficients of the finite difference method
def coefficients(d):
    
    # Use Diffusion Approximation for Disaster Risk
    lam_b_dx2 = (model.b / grid.dx) ** 2
    drift = model.mu - d - model.lam * model.b
    coe_1 = grid.dt * (0.5 * model.lam * lam_b_dx2 - 0.5 * drift / grid.dx)
    coe_2 = (1 - grid.dt * (model.delta + model.lam * lam_b_dx2)) * np.ones(grid.Nx + 1)
    coe_3 = grid.dt * (0.5 * model.lam * lam_b_dx2 + 0.5 * drift / grid.dx)
    return coe_1, coe_2, coe_3

# Define function for boundary condition at lower bound
def boundary_low(t):
    
    val = (-1 / model.delta * (1 - np.exp(-model.delta * (grid.tmax - t))) - np.exp(-model.delta * (grid.tmax - t))) / model.gam
    return val

# Initialize value function
V = np.zeros((grid.Nx + 1, grid.Nt + 1))
V[:, -1] = -np.exp(-model.gam * grid.x) / model.gam
V_ns_old = V[:, -1].copy()

# Initialie dividends
d = np.zeros_like(V)
d[:, -1] = policy(V[:, -1])
d_old = d[:, -1].copy()

# Solve HJB using finite differences
for j in reversed(range(grid.Nt)):
    t = j * grid.dt + grid.tmin
    coe_1, coe_2, coe_3 = coefficients(d_old)
    
    # Compue value function a previous time step
    V_ns = coe_2 * V_ns_old  + coe_3 * np.roll(V_ns_old , -1) + coe_1 * np.roll(V_ns_old, 1) - grid.dt * np.exp(-model.gam * d_old) / model.gam
    
    # Boundary value at lower bound
    V_ns[0] = boundary_low(t)
    
    # Linear extrapolation function at upper bound
    slope = (V_ns[-3] - V_ns[-2]) / (grid.x[-3] - grid.x[-2])
    intercept = V_ns[-2] - slope * grid.x[-2]
    V_ns[-1] = slope*grid.x[-1]+intercept
    
    # Save values
    V[:, j] = V_ns
    d[:, j] = policy(V_ns_old)
    
    # Set values for next iteration step
    V_ns_old = V[:, j].copy()
    d_old = d[:, j].copy()
    


print('Done Value Function Iteration')

# Monte Carlo simulation
print('Start Monte Carlo Iteration')
    
N_mc = 10000
grid_mc = np.arange(10, grid.Nt+1, 10)
plot_t = np.zeros(len(grid_mc)+1)
plot_t[1:] = grid.dt * grid_mc
prob = np.zeros((3, len(grid_mc)+1))
x_start = [1.5, 2.5, 5]

l=0
for x_0 in x_start:
    x_old = x_0 * np.ones(N_mc)
    print('Starting value x_0 ='+str(x_0))
    m = 1
    for n in range(grid.Nt+1):
        div = interp1d(grid.x, d[:, n], fill_value="extrapolate")(x_old)
        div[div <= 0]= 0
        eps = np.random.randn(N_mc)
        x_ns = (x_old + (model.mu - div - model.lam * model.b) * grid.dt +
                 np.sqrt(grid.dt * model.lam) * model.b * eps) * (x_old > 0)
        x_ns = x_ns * (x_ns > 0)
        x_old = x_ns
        if n in grid_mc:
             prob[l, m] = np.sum(x_ns == 0) / N_mc
             m=m+ 1
             
    l=l+1

# Plot results
plt.figure()
plt.plot(grid.x, V[:, 0], color='black', linewidth=2)
plt.title("Value function at $t=0$")
plt.xlabel("State $X_t$")
plt.ylabel("Value function $V(0, X_t)$")
plt.grid(False)
Vmin = V[:, 0].min()
plt.xlim([0, 10])
plt.ylim([Vmin, 0])

plt.figure()
plt.plot(grid.x, d[:, 0], color='black', linewidth=2)
plt.title("Dividends at $t=0$")
plt.xlabel("State $X_t$")
plt.ylabel("$d^{*}(0, X_t)$")
plt.grid(False)
dmax = d[:, 0].max()
plt.xlim([0, 10])
plt.ylim([0, dmax])

plt.figure()
colors = ['black', (0.5, 0.7, 0.2), (0.3, 0.2, 0.6)]
labels = ['$X_0 = 1.5$', '$X_0 = 2.5$', '$X_0 = 5.0$']
for i in range(3):
    plt.plot(plot_t, prob[i, :], color=colors[i], linewidth=2, label=labels[i])
plt.title("Ruin Probability $P(\\tau \\leq t)$")
plt.xlabel("Time $t$")
plt.ylabel("$P(\\tau \\leq t)$")
plt.legend()
plt.grid(False)
plt.xlim([0, plot_t[-1]])
tmax = plot_t.max()
plt.xlim([0, tmax])
plt.ylim([0, 1])
plt.show()



