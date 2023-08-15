# Production of dsRNA minicells
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint, solve_ivp
import sklearn

# ==================== Variables ==================== #
ecoli = 10000 # 10000 cells
dsRNA = 100
minicell = 0

free_dsRNA = dsRNA

ecoli_degrade_rate = 0.01 # 2/min
ecoli_replicate_rate = 0.2 # 20/min

ecoli_uptake_prob = 0.2 # 80% chance of uptake
ecoli_minicell_prob = 0.5 # 50% chance of minicell formation

ecoli_track = [ecoli]
dsRNA_track = [dsRNA]
minicell_track = [minicell]

empty_ecoli = ecoli
replicate_ecoli_with_ds = 0
untouched_ecoli_with_ds = 0

for min in range(100):
    # dsRNA + Ecoli
    uptaken_ecoli = ecoli_uptake_prob * empty_ecoli
    if uptaken_ecoli > free_dsRNA:
        uptaken_ecoli = free_dsRNA

    free_dsRNA -= uptaken_ecoli
    empty_ecoli -= uptaken_ecoli

    empty_ecoli = empty_ecoli * ecoli_replicate_rate - empty_ecoli * ecoli_degrade_rate + empty_ecoli * (1 - ecoli_degrade_rate - ecoli_replicate_rate)
    
    dead_ecoli_with_ds = uptaken_ecoli * ecoli_degrade_rate
    replicate_ecoli_with_ds += uptaken_ecoli * ecoli_replicate_rate
    untouched_ecoli_with_ds += uptaken_ecoli - dead_ecoli_with_ds - replicate_ecoli_with_ds

    minicell += replicate_ecoli_with_ds * ecoli_minicell_prob
    replicate_ecoli_with_ds -= replicate_ecoli_with_ds * ecoli_minicell_prob
    untouched_ecoli_with_ds += replicate_ecoli_with_ds * ecoli_minicell_prob

    free_dsRNA += dead_ecoli_with_ds

    minicell_track.append(minicell)
    dsRNA_track.append(free_dsRNA)
    ecoli_track.append(empty_ecoli + replicate_ecoli_with_ds + untouched_ecoli_with_ds)

    if free_dsRNA <= 0:
        break

# ==================== Plot ==================== #
plt.plot(minicell_track, label="minicell")
plt.plot(dsRNA_track, label="dsRNA")
plt.xlabel("Time (min)")
plt.ylabel("Concentration (cells/µL)")
plt.title("Minicell Production")
plt.legend()
plt.show(block=True)



