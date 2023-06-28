import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# The ETI model is modeling the ETI system in plants
# This model will modify the gamma value in the SIR model 
class ETImodel:
    def __init__(self, R, A):
        self.R = R
        self.A = A
        

if __name__ == "__main__":
    R = np.random.randint(30, 100) # Number of Resistence Genes