import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random

# ==================== Variables ==================== #
ml = 500
ecoli = 500 * ml    # cells/ml
minicell = 0
dsRNA = 0
siRNA = 0
risc = 0
tswv = 100
risc_tswv = 0

time = 500
time_step = 1
min = 0

limit = 1e11
limit_tswv = 1e15

start_infection = 0 # RISC will only start increasing after virus infection

# Rates & Percentages — Currently all made up values
ecoli_replicate_rate = np.log(2)/20   # double every 20 minutes convert to day
ecoli_degrade_rate = np.log(2)/6    # half life of 6 minutes convert to day
ecoli2dsRNA_percent = 0.05
dsRNA2siRNA_percent = 0.95
siRNA2risc_percent = 1.0 
tswv_replicate_rate = np.random.uniform(1, 1.5)
tswv_replicate_rate = 1.2
risc_tswv_degrade_percent = 1.000

# Stacks
ecoli_stack = []
minicell_stack = []
dsRNA_stack = []
siRNA_stack = []
risc_stack = []
tswv_stack = []
risc_tswv_stack = []

# ==================== ODE ==================== #
# Stopping
def stop(t, y, e_g, e_d, d_g, d_d, s_d, t_g, rt):
    return tswv > 1e15 or tswv < 1e-15

# ODE
def interference_ode(t, y, e_g, e_d, d_g, d_d, s_d, t_g, rt):
    global limit
    ecoli, dsRNA, siRNA, risc, tswv = y

    d_ecoli = ((e_g) * ecoli) * (1 - (ecoli/limit))     # Logistic Growth
    d_dsRNA = 500 * d_g * ecoli - d_d * dsRNA             # optimally 500 plasmid per minicell
    d_siRNA = 20 * d_d * dsRNA - s_d * siRNA              # 20 sites in dsRNA to become siRNA
    d_risc = s_d * siRNA                                  # 1 siRNA = 1 RISC 

    rt = 1 + 0.012 / 10 * (t // 5)

    if t > start_infection:
        half_life = np.log(2)/6.395             # half life of 6.395 minutes for mRNA
        diff = tswv - risc                      # difference between tswv and risc
        
        # if tswv more than risc by 20 percent
        if diff > risc:          
            d_tswv = - (risc) * 1.2
        else:
            tswv_2_risc = tswv/risc
            d_tswv = (t_g * tswv - half_life * tswv - risc * tswv_2_risc * rt) 
    else:
        d_tswv = 0     
    if tswv < 0:
        d_tswv = 0
    
    return [d_ecoli, d_dsRNA, d_siRNA, d_risc, d_tswv]


y0 = [ecoli, dsRNA, siRNA, risc, tswv]
t_span = (0, time)
rates = (ecoli_replicate_rate, ecoli_degrade_rate, ecoli2dsRNA_percent, dsRNA2siRNA_percent, siRNA2risc_percent, tswv_replicate_rate, risc_tswv_degrade_percent)

solution = solve_ivp(interference_ode, t_span, y0, args=(rates), dense_output=False, rtol=1e-3, atol=1e-6)

# Plot results
fig, ax = plt.subplots(3, 2, figsize=(20, 8))
fig.tight_layout(pad=4.0)
ax[0, 0].plot(solution.t, solution.y[0], label="Ecoli", color='blue')
ax[0, 0].set_title('Ecoli')
ax[0, 1].plot(solution.t, solution.y[1], label="dsRNA", color='orange')
ax[0, 1].set_title('dsRNA')
ax[1, 0].plot(solution.t, solution.y[2], label="siRNA", color='green')
ax[1, 0].set_title('siRNA')
ax[1, 1].plot(solution.t, solution.y[3], label="RISC", color='red')
ax[1, 1].set_title('RISC')
ax[2, 0].plot(solution.t, solution.y[4], label="TSWV", color='purple')
ax[2, 0].set_title('TSWV')
# This is to show the trend of the graphs
ax[2, 1].plot(solution.t, solution.y[0]/np.max(solution.y[0]), label="Ecoli")
ax[2, 1].plot(solution.t, solution.y[1]/np.max(solution.y[1]), label="dsRNA")
ax[2, 1].plot(solution.t, solution.y[2]/np.max(solution.y[2]), label="siRNA")
ax[2, 1].plot(solution.t, solution.y[3]/np.max(solution.y[3]), label="RISC")
ax[2, 1].plot(solution.t, solution.y[4]/np.max(solution.y[4]), label="TSWV")
ax[2, 1].set_title('Everything with respect to their percetage')    
ax[2, 1].legend(loc="upper left")
fig.suptitle('mRNA Interference')
plt.show()

