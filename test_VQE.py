'''
In this file I test that VQE works properly
'''

import numpy as np
from SA_VQE import SA_VQE_expec_val
import functions

dim_grid = (2,5)
seed = 42
J_right, J_down = functions.generate_couplings(dim_grid, seed)
hamiltonian = functions.hamiltonian(dim_grid, J_right, J_down)
obs_name = 'corr01'
obs = functions.observable(dim_grid, obs_name)

# Obtain ground state property by diagonalization
obs_diag, ground_state_energy_diag = functions.ground_state_expectation_value(dim_grid, hamiltonian, obs)
print(f"Expectation value of {obs_name} [Diagonalization]= {obs_diag}")
print(f'Ground state energy [Diagonalization] = {ground_state_energy_diag}')

# Obtain ground state property by SA VQE
depth = 3
opt_steps = 1000
learning_rate = 0.01
obs_SA_VQE, ground_state_energy_VQE = SA_VQE_expec_val(dim_grid, hamiltonian, obs, depth, opt_steps, learning_rate)
print(f"Expectation value of {obs_name} [VQE]= {obs_SA_VQE}")
print(f'Ground state energy [VQE] = {ground_state_energy_VQE}')
