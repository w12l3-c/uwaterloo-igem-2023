import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define ODEs
def system(t, y, dicer_processing_rate, risc_formation_rate, risc_degradation_rate):
    dsRNA, siRNA, risc_complex = y
    
    d_dsRNA = -dicer_processing_rate * dsRNA
    d_siRNA = dicer_processing_rate * dsRNA - risc_formation_rate * siRNA
    d_risc_complex = risc_formation_rate * siRNA - risc_degradation_rate * risc_complex
    
    return [d_dsRNA, d_siRNA, d_risc_complex]

# Parameters
dicer_processing_rate = 0.1
risc_formation_rate = 0.05
risc_degradation_rate = 0.02

# Initial conditions
dsRNA_0 = 100
siRNA_0 = 0
risc_complex_0 = 0

y0 = [dsRNA_0, siRNA_0, risc_complex_0]

# Time span
t_span = (0, 100)

# Solve ODEs
solution = solve_ivp(system, t_span, y0, args=(dicer_processing_rate, risc_formation_rate, risc_degradation_rate))

# Plot results
plt.plot(solution.t, solution.y[0], label="dsRNA")
plt.plot(solution.t, solution.y[1], label="siRNA")
plt.plot(solution.t, solution.y[2], label="RISC Complex")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Modeling dsRNA, siRNA, and RISC Concentrations Over Time")
plt.legend()
plt.show()
