'''
In this file we define the parameters used in the simulation
'''

# General parameters
dim_grid = (2,2)
obs_name = 'correlation'
qubits = ((0,0), (0,1))
num_examples = 100

# Parameters of VQE
depth = 3
opt_steps = 500
learning_rate = 0.01