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

time = 10
time_step = 1
day = 0

limit = 1e10
limit_tswv = 1e10

start_infection = 5 # RISC will only start increasing after virus infection

# Rates & Percentages — Currently all made up values
ecoli_replicate_rate = np.log(2)/20*60*24   # double every 20 minutes convert to day
ecoli_degrade_rate = np.log(2)/6*60*24    # half life of 6 minutes convert to day
ecoli2dsRNA_percent = 0.05
dsRNA2siRNA_percent = 0.95
siRNA2risc_percent = 1.0 
tswv_replicate_rate = np.random.uniform(0.1, 1) * 60 * 24
risc_tswv_degrade_percent = 0.0000001

tswv_cycle = 20
tswv_multiplier = 6000 # Average Range Negative Strand RNA burst size

# Stacks
ecoli_stack = []
minicell_stack = []
dsRNA_stack = []
siRNA_stack = []
risc_stack = []
tswv_stack = []
risc_tswv_stack = []

# ==================== ODE ==================== #

# Probability of RISC inferencing TSWV
# def risc_tswv_prob():
#     return random.uniform(0,1)

# Monte Carlo between TSWV and RISC
# def model_risc_tswv(risc, tswv):
#     tswv_change = 0
#     max_interaction = min(risc, tswv)

#     for i in range(max_interaction):
#         if random.uniform(0,1) < risc_tswv_prob():
#             tswv_change -= 1

#     return tswv_change

# Virus duplicate
# def model_tswv(tswv, t):
#     tswv_multiplier = random.randint(3000, 6000)
#     tswv_multipler = tswv_multiplier * (np.sin(2 * np.pi * t / tswv_cycle))
#     tswv += tswv * 0.2 * tswv_multipler # 0.2 percent of tswv replicate
#     return tswv

# Stopping
def stop(t, y, e_g, e_d, d_g, d_d, s_d, t_g, rt):
    return tswv > 1e15 or tswv < 1e-15

# ODE
def interference_ode(t, y, e_g, e_d, d_g, d_d, s_d, t_g, rt):
    global limit
    ecoli, dsRNA, siRNA, risc, tswv = y

    d_ecoli = ((e_g) * ecoli) * (1 - (ecoli/limit))     # Logistic Growth
    # d_minicell = e_d * ecoli * - m_d * minicell           # minicell touches plant cell it degrades instantly
    d_dsRNA = 500 * d_g * ecoli - d_d * dsRNA             # optimally 500 plasmid per minicell
    d_siRNA = 20 * d_d * dsRNA - s_d * siRNA              # 20 sites in dsRNA to become siRNA
    d_risc = s_d * siRNA                                  # 1 siRNA = 1 RISC - RISC bind with TSWV

    if t > start_infection:
        half_life = np.log(2)/6.395 * 24 * 60             # half life of 6.395 minutes for mRNA
        diff = tswv - risc
        if diff > 0:
            rt = risc
        else:
            rt = tswv
        d_tswv = (t_g * tswv - half_life * tswv - rt) * (1 - (tswv/limit_tswv))
    else:
        d_tswv = 0     
    
    return [d_ecoli, d_dsRNA, d_siRNA, d_risc, d_tswv]


y0 = [ecoli, dsRNA, siRNA, risc, tswv]
t_span = (0, time)
rates = (ecoli_replicate_rate, ecoli_degrade_rate, ecoli2dsRNA_percent, dsRNA2siRNA_percent, siRNA2risc_percent, tswv_replicate_rate, risc_tswv_degrade_percent)

solution = solve_ivp(interference_ode, t_span, y0, args=(rates), dense_output=False)

# Plot results
fig, ax = plt.subplots(3, 2, figsize=(20, 8))
fig.tight_layout(pad=4.0)
ax[0, 0].plot(solution.t, solution.y[0], label="Ecoli")
ax[0, 0].set_title('Ecoli')
ax[0, 1].plot(solution.t, solution.y[1], label="dsRNA")
ax[0, 1].set_title('dsRNA')
ax[1, 0].plot(solution.t, solution.y[2], label="siRNA")
ax[1, 0].set_title('siRNA')
ax[1, 1].plot(solution.t, solution.y[3], label="RISC")
ax[1, 1].set_title('RISC')
ax[2, 0].plot(solution.t, solution.y[4], label="TSWV")
ax[2, 0].set_title('TSWV')
# ax[2, 1].plot(solution.t, solution.y[5], label="RISC-TSWV")
# ax[2, 1].set_title('RISC-TSWV')
fig.suptitle('mRNA Interference')
plt.show()

