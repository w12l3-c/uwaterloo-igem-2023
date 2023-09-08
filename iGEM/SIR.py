import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import pandas as pd
import datetime
from matplotlib.animation import FuncAnimation

# SIR model for Tomato Spot Wilt Virus (TSWV) in plants

# Total population, N.
N = 1000000
I0 = 10  # Initial Infected Population
R0 = 0    # Initial Recovered Population
S0 = N - I0 - R0   # Initial Suspectible Population

# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta, gamma = 0.5, 1./10

# A grid of time points (in days)
days = 200
t = np.linspace(0, days, days)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N   # Suspectible
    dIdt = beta * S * I / N - gamma * I  # Infected
    dRdt = gamma * I # Recovered
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0

# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

def plotSIR(t, S, I, R, days):
    y = np.vstack([I/1000, S/1000, R/1000])
    fig, ax = plt.subplots()
    labels = ['Infected', 'Suspectible', 'Recovered']
    color_map = ["#4287f5", "#fcbd56", "#70deff"]
    ax.stackplot(t, y, labels=labels, colors=color_map)
    ax.set_title('SIR Model for TSWV')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number (1000s)')
    ax.set_ylim(0,1000)
    ax.set_xlim(0,days)
    ax.legend(loc='upper left')
    plt.show(block=True)

    # # Stacked graph in seaborn
    # df = pd.DataFrame({'Suspectible':S, 'Infected':I, 'Recovered':R})
    # df = df.melt(var_name='groups', value_name='vals')
    # g = sb.relplot(x='groups', y='vals', hue='groups', kind='line', data=df)
    # g.fig.autofmt_xdate()


plotSIR(t, S, I, R, days)


