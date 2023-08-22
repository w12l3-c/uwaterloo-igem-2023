import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sklearn

# ==================== Variables ==================== #
ecoli_with_ds = 1000
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

start_infection = 10

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

while tswv > 0 and sec < 500:
    # Ecoli Replication
    ecoli_with_ds_replicate = ecoli_with_ds * ecoli_replicate_percent  # only replicate ecoli with dsRNA can make minicell
    ecoli_with_ds -= ecoli_with_ds * ecoli_degrade_percent # deduct dead ecoli
    ecoli_with_ds -= ecoli_with_ds_replicate # deduct replicated ecoli

    # Ecoli to Minicell
    minicell += ecoli_with_ds_replicate * ecoli2minicell_percent
    single_ecoli_with_ds = ecoli_with_ds_replicate * ecoli2minicell_percent # ecoli that became minicell
    double_ecoli_with_ds = (ecoli_with_ds_replicate - single_ecoli_with_ds) * 2 # ecoli simply replicated
    ecoli_with_ds += single_ecoli_with_ds + double_ecoli_with_ds # add replicated ecoli back to ecoli

    ecoli_stack.append(ecoli_with_ds)

    # Minicell to dsRNA
    dsRNA += minicell * minicell2dsRNA_percent  # minicell burst into dsRNA
    minicell -= minicell * minicell2dsRNA_percent   # minicell is gone

    minicell_stack.append(minicell)

    # dsRNA to siRNA
    siRNA += dsRNA * dsRNA2siRNA_percent    # dsRNA is degraded into siRNA
    dsRNA -= dsRNA * dsRNA2siRNA_percent    # dsRNA is gone

    dsRNA_stack.append(dsRNA)

    # siRNA to RISC
    risc += siRNA * siRNA2risc_percent  # siRNA is degraded into RISC
    siRNA -= siRNA * siRNA2risc_percent # siRNA is gone

    siRNA_stack.append(siRNA)

    # Start of TSWV Infection
    if sec > start_infection:
        # mRNA Interferance
        risc_tswv += risc * initial_risc_tswv_prob  # Increase of risc bind with tswv
        risc -= risc * initial_risc_tswv_prob   # Decrease of free risc

        tswv -= risc_tswv # Decrease of tswv because risc bind with tswv
        if sec % tswv_cycle == 0 and sec != start_infection:
            tswv *= tswv_multiplier  # Burst of tswv

        released_risc = risc_tswv * risc_tswv_degrade_percent   # Released risc that clipped tswv (I am using method one not the hairpin)
        risc_tswv -= released_risc  # Decrease of risc + tswv because risc clipped tswv
        risc += released_risc   # Increase of free risc

        # Ajustment of initial_risc_tswv_prob
        # The more tswv the more likely risc will be tswv
        initial_risc_tswv_prob = np.log(tswv) / 1000
        if initial_risc_tswv_prob > 1:
            initial_risc_tswv_prob = 1.0

    # Increase in time
    sec += time_step
    
    risc_stack.append(risc)
    tswv_stack.append(tswv)
    risc_tswv_stack.append(risc_tswv)

    # Print
    print('Time: ', sec)
    print('Ecoli: ', ecoli_stack[-1])
    print('Minicell: ', minicell_stack[-1])
    print('dsRNA: ', dsRNA_stack[-1])
    print('siRNA: ', siRNA_stack[-1])
    print('RISC: ', risc_stack[-1])
    print('TSWV: ', tswv_stack[-1])
    print('RISC-TSWV: ', risc_tswv_stack[-1])
    

# ==================== Plotting ==================== #
# Plotting into multiply subplots
fig, ax = plt.subplots(3, 2, figsize=(20, 8))
ax[0, 0].plot(ecoli_stack)
ax[0, 0].set_title('Ecoli')
ax[0, 1].plot(minicell_stack)
ax[0, 1].set_title('Minicell')
ax[1, 0].plot(dsRNA_stack)
ax[1, 0].set_title('dsRNA')
ax[1, 1].plot(siRNA_stack)
ax[1, 1].set_title('siRNA')
ax[2, 0].plot(risc_stack)
ax[2, 0].set_title('RISC')
ax[2, 1].plot(tswv_stack)
ax[2, 1].set_title('TSWV')
fig.suptitle('mRNA Interference')
plt.show()

