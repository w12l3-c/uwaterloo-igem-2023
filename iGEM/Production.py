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

ecoli_degrade_rate = 2 # 2/min
ecoli_replicate_rate = 20 # 20/min

ecoli_uptake_prob = 0.8 # 80% chance of uptake
ecoli_minicell_prob = 0.5 # 50% chance of minicell formation

ecoli_track = [ecoli]
dsRNA_track = [dsRNA]
minicell_track = [minicell]

for min in range(100):
    ecoli = ecoli + ecoli_replicate_rate * ecoli - ecoli_degrade_rate * ecoli
    
    uptaken_dsRNA = ecoli_uptake_prob * free_dsRNA
    free_dsRNA = dsRNA - uptaken_dsRNA

    dsRNA = free_dsRNA + 2 * uptaken_dsRNA

    minicell = minicell + uptaken_dsRNA * ecoli_minicell_prob
    