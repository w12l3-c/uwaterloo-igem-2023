import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sklearn
import random

# ==================== Variables ==================== #
ml = 500
ecoli_with_ds = 500 * ml    # cells/ml
ecoli_with_ds_replicate = 0
minicell = 0
dsRNA = 0
siRNA = 0
risc = 0
tswv = 1
risc_tswv = 0

time = 100
time_step = 1
sec = 0

start_infection = 0 # RISC will only start increasing after virus infection

# Rates & Percentages — Currently all made up values
ecoli_replicate_percent = 0.5
ecoli_degrade_percent = 0.1
ecoli2minicell_percent = 0.5
minicell2dsRNA_percent = 0.9
dsRNA2siRNA_percent = 0.95
siRNA2risc_percent = 0.95

initial_risc_tswv_prob = 0.0    # The more tswv the more likely risc will be tswv
risc_tswv_degrade_percent = 0.9

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
# Modify the Growth Rates
def model_gr(rates):

    ...

# Modify the Degradation Rates
def model_dr(rates):

    ...

# Probability of RISC inferencing TSWV
def risc_tswv_prob():
    return random.uniform(0,1)

# Monte Carlo between TSWV and RISC
def model_risc_tswv(risc, tswv):
    tswv_change = 0
    max_interaction = min(risc, tswv)

    for i in range(max_interaction):
        if random.uniform(0,1) < risc_tswv_prob():
            tswv_change -= 1

    return tswv_change

# Virus duplicate
def model_tswv(tswv, t):
    tswv_multiplier = random.randint(3000, 6000)
    tswv_multipler = tswv_multiplier * (np.sin(2 * np.pi * t / tswv_cycle))
    tswv += tswv * 0.2 * tswv_multipler # 0.2 percent of tswv replicate
    return tswv

# Stopping
def stop(tswv):
    return tswv > 1e15 or tswv < 1e-15

def interference_ode(t, y, rates):
    ecoli, minicell, dsRNA, siRNA, risc, tswv = y
    e_g, e_d, m_d, d_d, s_d, t_d = rates    # Ecoli gr, ecoli dr, minicell dr, dsRNA dr, siRNA dr, tswv dr
    
    d_ecoli = e_g * ecoli - e_d * ecoli
    d_minicell = e_d * ecoli * - m_d * minicell
    d_dsRNA = m_d * minicell * - d_d * dsRNA
    d_siRNA = d_d * dsRNA * - s_d * siRNA
    d_risc = s_d * siRNA 

    if t > start_infection:
        d_tswv = model_tswv(tswv, t) + model_risc_tswv(risc, tswv) - t_d * tswv
    else:
        d_tswv = 0
    
    return [d_ecoli, d_minicell, d_dsRNA, d_siRNA, d_risc, d_tswv]


# solve_ivp(interference_ode, t_span, y0, args=(rates), dense_output=True, events=stop)