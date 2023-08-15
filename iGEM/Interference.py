import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint, solve_ivp
import sklearn

# ==================== Constants ==================== #
TIME_STEP = 1 # minute
TOTAL_TIME = 10 # minute

INITIAL_MS_CONCENTRATION = 1.0 # cells/µL (unit TBD)
MS_BREAKDOWN_RATE = 0.1 # 1/min

RISC_FORMATION_RATE = 1 # 1/min
INITIAL_RISC_CONCENTRATION = 0.0 # cells/µL

INFECTION_START_TIME = 5 # min
MRNA_FORMATION_RATE = 2 # 1/min
MRNA_DISSOCIATION_RATE = 0.5 # 1/min
INITIAL_MRNA_CONCENTRATION = 1.0 # cells/µL

PROBABILITY_OF_INTERFERENCE = 0.5 # Inference probability of RISC and viral mRNA

# If we know the sequence of mRNA and RISC then this prob would be a calculated

# ==================== Pipeline ==================== #
# minicell -> dsRNA -> Dicer -> siRNA -> RISC -> mRNA inference + mRNA degradation
# rn assume dsRNA -> Dicer -> siRNA is instantaneous

# ==================== Variables ==================== #
ms_concentration = INITIAL_MS_CONCENTRATION

free_RISC_concentration = INITIAL_RISC_CONCENTRATION
occupied_RISC_concentration = 0.0   

free_mRNA_concentration = INITIAL_MRNA_CONCENTRATION
occupied_mRNA_concentration = 0.0


def create_RISC(siRNA_concentration):
    if siRNA_concentration > 0:
        RISC_concentration = RISC_FORMATION_RATE * siRNA_concentration
        siRNA_concentration -= RISC_FORMATION_RATE * siRNA_concentration
    return RISC_concentration, siRNA_concentration

RISC_stack = []
free_mRNA_concentration_stack = []

for min in range(1, TOTAL_TIME+1, TIME_STEP):
    ms_concentration -= MS_BREAKDOWN_RATE * ms_concentration 
    dsRNA_concentration = INITIAL_MS_CONCENTRATION - ms_concentration
    dicer_concentration = dsRNA_concentration   # TBD
    siRNA_concentration = dicer_concentration   # TBD

    RISC_concentration, siRNA_concentration = create_RISC(siRNA_concentration) 

    if min > INFECTION_START_TIME:
        free_mRNA_concentration += MRNA_FORMATION_RATE * free_mRNA_concentration

        if free_RISC_concentration > 0:
            free_RISC_concentration -= RISC_concentration * PROBABILITY_OF_INTERFERENCE
            occupied_RISC_concentration += RISC_concentration * PROBABILITY_OF_INTERFERENCE

            free_mRNA_concentration -= occupied_RISC_concentration 
            occupied_mRNA_concentration += occupied_RISC_concentration
            
        occupied_mRNA_concentration -= MRNA_DISSOCIATION_RATE * occupied_mRNA_concentration
        free_mRNA_concentration += occupied_RISC_concentration - occupied_mRNA_concentration
        occupied_RISC_concentration = occupied_mRNA_concentration

    print("Minute: ", min)
    print("RISC: ", RISC_concentration)
    print("mRNA: ", free_mRNA_concentration)

    RISC_stack.append(RISC_concentration)
    free_mRNA_concentration_stack.append(free_mRNA_concentration)

# ==================== Plotting ==================== #
plt.plot(range(1, TOTAL_TIME+1, TIME_STEP), RISC_stack, label="RISC")
plt.plot(range(1, TOTAL_TIME+1, TIME_STEP), free_mRNA_concentration_stack, label="mRNA")
plt.xlabel("Time (min)")
plt.ylabel("Concentration (cells/µL)")
plt.title("Stochastic Model of Minicell Formation")
plt.show(block=True)
            
        





