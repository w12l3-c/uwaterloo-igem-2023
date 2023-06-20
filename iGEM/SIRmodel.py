import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from tqdm import tqdm

class SIRmodel:
    def __init__(self, N, I0, R0, S0, Rn, beta, gamma, days):
        self.N = N    # Total population
        self.I0 = I0    # Initial Infected Population
        self.R0 = R0    # Initial Recovered Population
        self.S0 = S0    # Initial Suspectible Population
        self.Rn = Rn    # Custom Secondary Infection Rn > 1 -> Create more infections
        self.bRn = beta/gamma # The best value for Rn
        self.beta = beta    # contact rate, beta, (in 1/days).
        self.gamma = gamma  # mean recovery rate, gamma, (in 1/days).
        self.days = days
        self.t = np.linspace(0, days, days)
        self.y0 = S0, I0, R0

        if self.N >= 1e8:
            self.ratio = 1e6
        elif self.N >= 1e5:
            self.ratio = 1e3
        else:
            self.ratio = 1

    ## ODE methof for SIR model
    def deriv(self, y, t, N, beta, gamma):
        
        S, I, R = y
        self.bRn = self.bRn * S / N
        self.dSdt = -beta * S * I / N   
        self.dIdt = beta * S * I / N - gamma * I  
        self.dRdt = gamma * I 
        return self.dSdt, self.dIdt, self.dRdt

    def fit_odeint(self):
        self.S, self.I, self.R = odeint(self.deriv, self.y0, self.t, args=(self.N, self.beta, self.gamma)).T
        
    def plotSIR(self):
        y = np.vstack([self.I/self.ratio, self.S/self.ratio, self.R/self.ratio])
        fig, ax = plt.subplots()
        labels = ['Infected', 'Suspectible', 'Recovered']
        color_map = ["#4287f5", "#fcbd56", "#70deff"]
        ax.stackplot(self.t, y, labels=labels, colors=color_map)
        ax.set_title(f'SIR Model for TSWV with Transmit Rate of {self.beta:.2f} and Recovery Rate of {self.gamma:.4f}')
        ax.set_xlabel('Time /days')
        ax.set_ylabel(f'Number in {self.ratio:.0f}s')
        ax.set_ylim(0,self.N/self.ratio)
        ax.set_xlim(0,self.days)
        ax.legend(loc='upper left')
        plt.show(block=True)

    ## Monte Carlo Simulation



if __name__ == '__main__':
    N = 1e9
    I0 = 10
    R0 = 0
    S0 = N - I0 - R0

    beta = 0.5
    gamma = 1./10
    Rn = 2.5

    print("How many trails do you want to run?")
    trails = int(input("Enter a number:"))

    for i in tqdm(range(trails)):
        TSWV = SIRmodel(N=N, I0=I0, R0=R0, S0=S0, Rn=Rn, beta=beta, gamma=gamma, days=80)
        TSWV.fit_odeint()
        TSWV.plotSIR()

        beta += 0.1
        gamma /= 10
        
