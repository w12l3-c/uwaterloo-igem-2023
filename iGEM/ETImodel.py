import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# The ETI model is modeling the ETI system in plants
# This model will modify the gamma value in the SIR model 
class PlantETI:
    def __init__(self, r_gene_frequency):
        self.r_gene_frequency = r_gene_frequency
        self.beta = 0.4  # Infection rate
        self.gamma_normal = 0.1  # Recovery rate for non-resistant plants
        self.gamma_resistant = 0.05  # Recovery rate for resistant plants
        self.infected = False

    def infect(self):
        self.infected = True

    def recover(self):
        self.infected = False

    def update_recovery_rate(self):
        if self.infected and self.has_effective_r_gene():
            self.gamma = self.gamma_resistant
        else:
            self.gamma = self.gamma_normal

    def has_effective_r_gene(self):
        # Determine if the plant has an effective R gene based on the frequency
        # Return True if a random number is less than the R gene frequency
        return np.random.random() < self.r_gene_frequency

    def simulate_infection(self):
        # Simulate the infection process
        self.infect()
        self.update_recovery_rate()

        # Run the simulation for a certain duration or until the plant recovers
        time = 0
        while self.infected:
            # Perform necessary computations for the ETI model

            # Update the recovery rate based on the presence of an effective R gene
            self.update_recovery_rate()

            # Increment time
            time += 1

        return time


if __name__ == "__main__":
    R = np.random.randint(30, 100) # Number of Resistence Genes