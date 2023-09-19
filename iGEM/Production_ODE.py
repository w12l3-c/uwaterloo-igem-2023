import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def system(t, y, ecoli_uptake_rate, minicell_formation_rate):
    ecoli, dsRNA, minicell = y
    
    uptaken_ecoli = ecoli * ecoli_uptake_rate
    
    d_ecoli = -uptaken_ecoli
    d_dsRNA = uptaken_ecoli - minicell_formation_rate * uptaken_ecoli
    d_minicell = uptaken_ecoli * minicell_formation_rate
    
    return [d_ecoli, d_dsRNA, d_minicell]

# Parameters
ecoli_uptake_rate = 0.1
minicell_formation_rate = 0.05

# Initial conditions
ecoli_0 = 10000
dsRNA_0 = 100
minicell_0 = 0

y0 = [ecoli_0, dsRNA_0, minicell_0]

t_span = (0, 100)

solution = solve_ivp(system, t_span, y0, args=(ecoli_uptake_rate, minicell_formation_rate))

free_dsRNA = solution.y[1] - solution.y[2]


plt.plot(solution.t, solution.y[0], label="E. coli")
plt.plot(solution.t, free_dsRNA, label="Free dsRNA")
plt.plot(solution.t, solution.y[2], label="Minicell")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Stochastic Model of Minicell Formation (ODE)")
plt.legend()
plt.show()
