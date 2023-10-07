import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==== ODE System ==== #
# *Note: Defining for modelling dsrna-containing minicell production
# *Cite: Bergstrom et al., 2003; Neofytou, 2017
def mc_prod_system(t, y, ecoli_replicate_rate, ecoli_degrade_rate, ecoli_carrying_capacity, mc_form_rate, dsrna_conversion_factor, dsrna_prod_rate, dsrna_degrade_rate, sirna_prod_rate, sirna_degrade_rate, risc_bind_rate, risc_dissociate_rate, mrna_sil_rate, mrna_degrade_rate, mrna_replenish_rate):
    
    ecoli, mc, dsrna, sirna, risc, target_mrna = y

    # Logistic growth of E. coli population
    ecoli_growth = ecoli_replicate_rate * ecoli * \
        (1 - ecoli / ecoli_carrying_capacity)

    # ODEs for production of dsRNA-containing minicells
    d_ecoli = ecoli_growth - ecoli_degrade_rate * ecoli - mc_form_rate * ecoli
    d_minicell = mc_form_rate * ecoli
    d_dsrna = mc_form_rate * ecoli * dsrna_conversion_factor - \
        dsrna_degrade_rate * dsrna - dsrna_prod_rate * mc

    # ODEs for RNAi mechanisms within plant cells
    d_sirna = sirna_prod_rate - sirna_degrade_rate * \
        sirna - risc_bind_rate * sirna * risc
    d_risc = risc_bind_rate * dsrna - risc_dissociate_rate * risc
    d_target_mrna = mrna_replenish_rate - mrna_degrade_rate * \
        target_mrna - mrna_sil_rate * risc * target_mrna

    return [d_ecoli, d_minicell, d_dsrna, d_sirna, d_risc, d_target_mrna]

# ==== Parameters ==== #
# *Cite: Bergstrom et al., 2003; Voloudakis et al., 2014
ecoli_replicate_rate = 0.2  # 20 cells/min
ecoli_degrade_rate = 0.01  # 1 cell/min
ecoli_carrying_capacity = 20000  # 20k cells  

mc_form_rate = 0.05  # Rate of dsRNA-containing minicell formation (5% of E. coli)
dsrna_conversion_factor = 1.0  # 100 dsRNA/minicell

dsrna_prod_rate = 0.05  # 5% of E. coli
dsrna_degrade_rate = 0.1  # 10% of dsRNA

sirna_prod_rate = 0.05  # 5% of dsRNA
sirna_degrade_rate = 0.02  # 2% of siRNA

risc_bind_rate = 0.3  # Rate of RISC binding to siRNA (30% of siRNA)
risc_dissociate_rate = 0.1  # 10% of RISC

mrna_sil_rate = 0.2  # Rate of gene silencing by RISC (20% of RISC)
mrna_degrade_rate = 0.2  # 20% of mRNA
mrna_replenish_rate = 0.15  # viral (TSWV) mRNA synthesis (15% of mRNA)

# ==== Initial Conditions ==== #
# *Note: Concentrations in cells/µL, excluding E. coli (in #)
# *Cite: Bergstrom et al., 2003; Voloudakis et al., 2014
ecoli_0 = 10000  
mc_0 = 10  
dsrna_0 = 100  
sirna_0 = 0.0 
risc_0 = 0.0 
target_mrna_0 = 10.0 

y0 = [ecoli_0, mc_0, dsrna_0, sirna_0, risc_0, target_mrna_0]
t_span = (0, 100)  # Time span for simulation (1000 min)

# Solve ODE system
sol = solve_ivp(mc_prod_system, t_span, y0, args=(ecoli_replicate_rate, ecoli_degrade_rate, ecoli_carrying_capacity, mc_form_rate, dsrna_conversion_factor,
    dsrna_prod_rate, dsrna_degrade_rate, sirna_prod_rate, sirna_degrade_rate, risc_bind_rate, risc_dissociate_rate, mrna_sil_rate, mrna_degrade_rate, mrna_replenish_rate), vectorized=True, rtol=1e-3, atol=1e-6)

# Calculate free dsRNA (i.e., not in minicells)
free_dsrna = sol.y[1] - sol.y[2]

# ==== Plot Results ==== #
plt.plot(sol.t, sol.y[0], label="E. coli")
plt.plot(sol.t, free_dsrna, label="Free dsRNA")
plt.plot(sol.t, sol.y[2], label="Minicell")
plt.xlabel("Time (min)")
plt.ylabel("Concentration (cells/µL)")
plt.title("Stochastic Model of Minicell Formation (ODE)")
plt.legend()
plt.show()

# ==== References ==== #
# Bergstrom, C. T., McKittrick, E., &amp; Antia, R. (2003). Mathematical models of RNA silencing: Unidirectional amplification limits accidental self-directed reactions. Proceedings of the National Academy of Sciences, 100(20), 11511–11516. https://doi.org/10.1073/pnas.1931639100 
# Neofytou, G. (2017). Mathematical models of RNA interference in plants. Sussex Research Online. 
# Voloudakis, A. E., Holeva, M. C., Sarin, L. P., Bamford, D. H., Vargas, M., Poranen, M. M., &amp; Tenllado, F. (2014). Efficient double-stranded RNA production methods for utilization in plant virus control. Methods in Molecular Biology, 255–274. https://doi.org/10.1007/978-1-4939-1743-3_19 

