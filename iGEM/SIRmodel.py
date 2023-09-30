import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize
from tqdm import tqdm

class SIRmodel:
    def __init__(self, N, I0, R0, S0, Rn, beta, gamma, delta, days):
        self.N = N      # Total population
        self.I0 = I0    # Initial Infected Population
        self.R0 = R0    # Initial Recovered Population
        self.S0 = S0    # Initial Suspectible Population
        self.D0 = 0     # Initial Dead Population
        
        self.Rn = Rn    # Base Secondary Infection Rn > 1 -> Create more infections
        self.Rt = 0     # Real-time Secondary Infection
        # self.R_exp = beta/gamma # Expected Secondary Infection
        self.RnCheck = []
        
        self.beta = beta    # contact rate, beta, (in 1/days).
        self.gamma = gamma  # mean recovery rate, gamma, (in 1/days).
        self.delta = delta   # death rate, delta, (in 1/days).
        self.days = days
        
        self.t_span = (0, days)                 # (start, max days)
        self.t_eval = [x for x in range(days)]  # list of the days
        self.t = np.linspace(0, days, days)     # (0, amount of days, steps)
        self.endtime = days                     # endtime
        self.y0 = S0, I0, R0, D0                # Initial conditions vector

        self.fig, self.ax = plt.subplots()      # Create a figure and an axes.
        self.lines = []

        # Ratio for population
        if self.N >= 1e8:
            self.ratio = 1e6
        elif self.N >= 1e5:
            self.ratio = 1e3
        else:
            self.ratio = 1

    ## ODE method for SIR model
    def deriv(self, y, t, N, beta, gamma, delta):
        S, I, R, D = y
        self.Rt = self.Rn * S
        # Maximum number of infected people is when Rt = 1
        # Rt > 1 -> Increase Infected, Rt < 1 -> Decrease Infected
        # self.dIdt = gamma * I * (self.Rt - 1) / N

        # The Differentials
        self.dSdt = -beta * S * I / N                           # Suspectible population
        self.dIdt = beta * S * I / N - gamma * I - delta * I    # Infected population
        self.dRdt = gamma * I                                   # Recovered population
        self.dDdt = delta * I                                   # Dead population

        self.d2Idt2 = pow(gamma, 2) * pow((self.Rt - 1), 2) * I - gamma * beta * self.Rt * pow(I, 2) - gamma * I  
        self.RnCheck.append(self.Rt)

        return self.dSdt, self.dIdt, self.dRdt, self.dDdt

    def termination(self, t, y):
        return y[1] - 0 < 0.01 or y[1] - self.N < 0.01  # Terminate when I = 0 or I = N

    def ode(self):
        # Running the ODE solver with the deriv function
        self.S, self.I, self.R, self.D = odeint(self.deriv, self.y0, self.t, args=(self.N, self.beta, self.gamma, self.delta)).T
        
    def plotSIR(self):
        # This one kind of unused now because it only display one iteration of ODE
        # It only works when the recovery, infection and death rates are determined
        y = np.vstack([self.D/self.ratio, self.I/self.ratio, self.S/self.ratio, self.R/self.ratio])
        fig, ax = plt.subplots()
        labels = ['Deceased', 'Infected', 'Suspectible', 'Recovered']
        color_map = ["#808080", "#db1d0f", "#0e85ed", "#19e653"]
        # ax.stackplot(self.t, y, labels=labels, colors=color_map)
        ax.stackplot(self.t, y, labels=labels, colors=color_map)
        ax.set_title(f'SIR Model for TSWV with Transmit Rate of {self.beta:.2f} and Recovery Rate of {self.gamma:.4f}')
        ax.set_xlabel('Time in days')
        ax.set_ylabel(f'Population in {self.ratio:.0f}s')
        ax.set_ylim(0,self.N/self.ratio)
        ax.set_xlim(0,self.days)
        # ax.set_xlim(0, self.endtime[-1])
        ax.legend(loc='upper left')
        plt.show(block=True)

    ## Monte Carlo Simulation
    # Monte Carlo Simulation by my own logic
    def monteCarlo(self, num_simulations=9):
        self.results = []
        self.hyperparameters = []
        self.num_simulations = num_simulations
        for _ in tqdm(range(num_simulations)):
            S = [self.S0]
            I = [self.I0]
            R = [self.R0]
            D = [self.D0]
            t = [0]

            # Randomness Event
            self.beta = max(0.05, min(1.0, np.random.normal(0.525, 0.15)))
            self.gamma = max(0.1, min(0.2, np.random.normal(0.15, 0.025)))
            self.delta = max(0.01, min(0.05, np.random.normal(0.03, 0.01)))

            # Mimic SIR but the rates are randomized
            for day in range(self.days):
                new_infected = np.random.binomial(S[-1], self.beta * I[-1] / self.N)
                new_recovered = np.random.binomial(I[-1], self.gamma)
                new_deceased = np.random.binomial(I[-1], self.delta)

                S.append(S[-1] - new_infected)
                I.append(I[-1] + new_infected - new_recovered)
                R.append(R[-1] + new_recovered)
                D.append(D[-1] + new_deceased)
                t.append(day + 1)
            
            self.results.append((S, I, R, D, t))
            self.hyperparameters.append((self.beta, self.gamma, self.delta))

    # Monte Carlo Simulation with ODE
    def monteCarlo2(self, num_sim=100):
        self.results = []
        self.hyperparameters = []
        self.num_simulations = num_sim
        for _ in tqdm(range(num_sim)):
            # Randomness Event
            self.beta = np.random.uniform(0.04, 0.25)
            self.gamma = np.random.uniform(0, 0.05)
            self.delta = np.random.uniform(0, 1/21)

            self.hyperparameters.append((self.beta, self.gamma, self.delta))

            # Run the ODE solver with the deriv function
            self.y0 = self.S0, self.I0, self.R0, self.D0
            S, I, R, D = odeint(self.deriv, self.y0, self.t, args=(self.N, self.beta, self.gamma, self.delta)).T
            self.results.append((S, I, R, D, np.array(self.t_eval)))

    # Plot the Monte Carlo Simulation
    def plotMonteCarlo(self):
        self.results = np.array(self.results)
        self.hyperparameters = np.array(self.hyperparameters)

        self.mean_results = np.mean(self.results, axis=0)
        self.std_results = np.std(self.results, axis=0)

        self.mean_hyperparameters = np.mean(self.hyperparameters, axis=0)
        self.std_hyperparameters = np.std(self.hyperparameters, axis=0)

        # Extract all the arrays from the mean_results, mean_hyperparameters
        S, I, R, D, t = self.mean_results
        beta, gamma, delta = self.mean_hyperparameters
        # Calculate the R0 
        R0 = beta / gamma

        print(type(S))
        # Print the results
        print(f"Suspectible:{S[-1]:.0f} | Infected:{I[-1]:.0f} | Recovered:{R[-1]:.0f} | Deceased:{D[-1]:.0f}")

        # Plot the results
        y = np.vstack([D, I, S, R])
        labels = ['Deceased', 'Infected', 'Suspectible', 'Recovered']
        color_map = ["#808080", "#db1d0f", "#0e85ed", "#19e653"]
        plt.title(f'Monte Carlo Sim for TSWV with {self.num_simulations} simulations \nBeta:{beta:.2f} | Gamma:{gamma:.4f} | Delta:{delta:.4f} | R0:{R0:.2f}')
        plt.stackplot(t, y, labels=labels, colors=color_map)
        plt.xlabel('Time (days)')
        plt.ylabel(f'Population')
        plt.legend(loc='upper left')

        # Animation (Currently not working)
        # self.animateMonteCarlo(S, I, R, D, t, beta, gamma, delta, R0, labels, color_map)
        plt.show(block=True)

    # Update the plot for animation
    def animateMonteCarlo(self, S, I, R, D, t, beta, gamma, delta, R0, labels, color_map):
        fig, ax = plt.subplots()
        ax.set_title(f'Monte Carlo Sim for TSWV with {self.num_simulations} simulations \nBeta:{beta:.2f} | Gamma:{gamma:.4f} | Delta:{delta:.4f} | R0:{R0:.2f}')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel(f'Population')
        
        population = np.vstack([D, I, S, R])

        stacks, = ax.stackplot([], [[]*4], labels=labels, colors=color_map)
        ax.legend(loc='upper left')

        def animate(i):
            for stack, data_layer in zip(stacks, population):
                stack.set_x(t[:i])
                stack.set_y(data_layer[:i])
            return stacks

        anim = FuncAnimation(fig, animate, frames=len(t), interval=20, blit=True)
        
        plt.show(block=True)
        anim.save('SIRmodel.gif', writer='imagemagick')


if __name__ == '__main__':
    N = 1e4                # Total plant population
    D0 = 0                 # Initial Dead Population
    I0 = 10                # Initial Infected Population
    R0 = 0                 # Initial Recovered Population
    S0 = N - I0 - R0 - D0  # Initial Suspectible Population

    Rn = 1.5
    gamma_iRNA = 0.3    # Probability of plant to pick up immunity trait from iRNA
    
    # Iterate through Rn to find best Rt
    # Although technically best Rt = S*Rn*e^(Rn*(gamma*(N-R)*t - gamma^2*I*(t^2/2) + (1/Rn)*(S/I)*(e^(-gamma*Rn*I*t - 1)))))
    # But the solve_ivp will deal with that for us and iterate Rt
    
    print("How many sims do you want to run (default 100)?")
    trails = int(input("Enter a number:"))

    TSWV = SIRmodel(N=N, I0=I0, R0=R0, S0=S0, Rn=Rn, beta=0.4, gamma=0.07, delta=1/21, days=100)
    TSWV.monteCarlo2(trails)
    TSWV.plotMonteCarlo()
