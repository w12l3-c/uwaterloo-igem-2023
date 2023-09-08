import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize
from tqdm import tqdm
from ETImodel import PlantETI

class SIRmodel:
    def __init__(self, N, I0, R0, S0, Rn, beta, gamma, days):
        self.N = N      # Total population
        self.I0 = I0    # Initial Infected Population
        self.R0 = R0    # Initial Recovered Population
        self.S0 = S0    # Initial Suspectible Population
        
        self.Rn = Rn            # Base Secondary Infection Rn > 1 -> Create more infections
        self.Rt = 0             # Real-time Secondary Infection
        # self.R_exp = beta/gamma # Expected Secondary Infection
        self.RnCheck = []
        
        self.beta = beta    # contact rate, beta, (in 1/days).
        self.gamma = gamma  # mean recovery rate, gamma, (in 1/days).
        self.days = days
        
        self.t_span = (0, days) # (start, max days)
        self.t_eval = [x for x in range(days)]
        self.t = np.linspace(0, days, days) # (0, amount of days, steps)
        self.endtime = days
        self.y0 = S0, I0, R0

        self.fig, self.ax = plt.subplots()
        self.lines = []

        if self.N >= 1e8:
            self.ratio = 1e6
        elif self.N >= 1e5:
            self.ratio = 1e3
        else:
            self.ratio = 1

    ## ODE method for SIR model
    def deriv(self, y, t, N, beta, gamma):
        S, I, R = y
        self.Rt = self.Rn * S
        self.dSdt = -beta * S * I / N  
        # Maximum number of infected people is when Rt = 1
        # Rt > 1 -> Increase Infected, Rt < 1 -> Decrease Infected
        # self.dIdt = gamma * I * (self.Rt - 1) / N
        self.dIdt = beta * S * I / N - gamma * I
        self.d2Idt2 = pow(gamma, 2) * pow((self.Rt - 1), 2) * I - gamma * beta * self.Rt * pow(I, 2) - gamma * I    
        self.dRdt = gamma * I 
        self.RnCheck.append(self.Rt)
        return self.dSdt, self.dIdt, self.dRdt

    def termination(self, t, y):
        return y[1] - 0 < 0.01 or y[1] - self.N < 0.01  # Terminate when I = 0 or I = N

    def ode(self):
        self.S, self.I, self.R = odeint(self.deriv, self.y0, self.t, args=(self.N, self.beta, self.gamma)).T
        # print(self.RnCheck[-5:-1])

        # solution = solve_ivp(self.deriv, self.t_span, self.y0, args=(self.N, self.beta, self.gamma), events=self.termination)
        # print(solution)
        # self.endtime = solution.t
        # self.S, self.I, self.R = solution.y
        
    def plotSIR(self):
        y = np.vstack([self.I/self.ratio, self.S/self.ratio, self.R/self.ratio])
        fig, ax = plt.subplots()
        labels = ['Infected', 'Suspectible', 'Recovered']
        color_map = ["#db1d0f", "#0e85ed", "#19e653"]
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
            t = [0]

            # Randomness Event
            self.beta = max(0.05, min(1.0, np.random.normal(0.525, 0.15)))
            self.gamma = max(0.1, min(0.2, np.random.normal(0.15, 0.025)))
        
            # threshold = 8

            for day in range(self.days):
                new_infected = np.random.binomial(S[-1], self.beta * I[-1] / self.N)
                # if new_infected > threshold:
                #     new_infected = threshold
                #     threshold = np.random.randint(1, 10)

                new_recovered = np.random.binomial(I[-1], self.gamma)

                S.append(S[-1] - new_infected)
                I.append(I[-1] + new_infected - new_recovered)
                R.append(R[-1] + new_recovered)
                t.append(day + 1)
            
            self.results.append((S, I, R, t))
            self.hyperparameters.append((self.beta, self.gamma))

    # Monte Carlo Simulation with ODE
    def monteCarlo2(self, num_sim=100):
        self.results = []
        self.hyperparameters = []
        self.num_simulations = num_sim
        for _ in tqdm(range(num_sim)):
            # Randomness Event
            self.beta = np.random.uniform(0.5, 1.5)
            self.gamma = np.random.uniform(0, 0.2)

            self.hyperparameters.append((self.beta, self.gamma))

            self.y0 = self.S0, self.I0, self.R0
            S, I, R = odeint(self.deriv, self.y0, self.t, args=(self.N, self.beta, self.gamma)).T
            self.results.append((S, I, R, np.array(self.t_eval)))

    def plotMonteCarlo(self):
        self.results = np.array(self.results)
        self.hyperparameters = np.array(self.hyperparameters)

        self.mean_results = np.mean(self.results, axis=0)
        self.std_results = np.std(self.results, axis=0)

        self.mean_hyperparameters = np.mean(self.hyperparameters, axis=0)
        self.std_hyperparameters = np.std(self.hyperparameters, axis=0)

        S, I, R, t = self.mean_results
        beta, gamma = self.mean_hyperparameters
        y = np.vstack([I, S, R])
        labels = ['Infected', 'Suspectible', 'Recovered']
        color_map = ["#db1d0f", "#0e85ed", "#19e653"]
        plt.title(f'Monte Carlo Sim for TSWV with {self.num_simulations} simulations | beta:{beta:.2f} | gamma:{gamma:.4f}')
        plt.stackplot(t, y, labels=labels, colors=color_map)
        plt.xlabel('Time')
        plt.ylabel(f'Population')
        plt.legend(loc='upper left')
        
        self.ani = FuncAnimation(self.fig, self.update_plot, frames=len(t), blit=True)
        plt.show(blocj=True)

    def update_plot(self, frame):
        for i, line in enumerate(self.lines):
            line.set_data(self.mean_results[3][:frame], self.mean_results[i][:frame])
        return self.lines

            
# I'll add PTI and ETI later as it is correlated with the iRNA and lab


if __name__ == '__main__':
    N = 1e3
    I0 = 10
    R0 = 0
    S0 = N - I0 - R0

    beta = 0.5
    gamma = 0           # TSEV has 100% fatality rate
    gamma_iRNA = 0.3    # Probability of plant to pick up immunity trait from iRNA
    
    Rn = 1    # Iterate through Rn to find best Rt
    # Although technically best Rt = S*Rn*e^(Rn*(gamma*(N-R)*t - gamma^2*I*(t^2/2) + (1/Rn)*(S/I)*(e^(-gamma*Rn*I*t - 1)))))
    # But the solve_ivp will deal with that for us and iterate Rt

    # print("How many trails do you want to run?")
    # trails = int(input("Enter a number:"))

    # for i in tqdm(range(trails)):
    #     TSWV = SIRmodel(N=N, I0=I0, R0=R0, S0=S0, Rn=Rn, beta=beta, gamma=gamma, days=80)
    #     TSWV.ode()
    #     TSWV.plotSIR()

    #     beta += 0.1
    #     gamma /= 10
    
    print("How many sims do you want to run (default 9)?")
    trails = int(input("Enter a number:"))

    TSWV = SIRmodel(N=N, I0=I0, R0=R0, S0=S0, Rn=Rn, beta=0.4, gamma=0.05, days=150)
    TSWV.monteCarlo2(trails)
    TSWV.plotMonteCarlo()
