'''
Author: Cesar Hernando

In this file, we train a Machine Learning (ML) model to learn to predict properties of a 2D antiferromagnetic system.

'''

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
import functions
from SA_VQE import SA_VQE_expec_val

mode = 'basic_test'

if mode == 'basic_test':
    # Set the number of qubits in each row/column of the square grid
    dim_grid = (2,2)

    # Generate the coupling coefficients
    seed = 1
    J_right, J_down = functions.generate_couplings(dim_grid, seed)

    # Visualize the lattice with the randomly generated couplings
    visualize = False
    if visualize:
        functions.visualize_couplings(dim_grid, J_right, J_down)

    # Obtain the Hamiltonian (qml.Hamiltonian) of the 2D Antiferromagnetic lattice
    hamiltonian = functions.hamiltonian(dim_grid, J_right, J_down)

    # Define the observable by its name
    # Options: 'corr01', 'magnetization'
    obs_name = 'magnetization'
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


elif mode == 'training':
    dim_grid = (2,5)
    num_examples = 1000
    obs_name = 'corr01'

    X, y = functions.generate_training_set(dim_grid, num_examples, obs_name)

    verbose = False
    if verbose:
        plt.figure()
        plt.plot(range(len(y)), y)
        plt.xlabel('Coupling random realization')
        plt.ylabel(f'Expectarion value of {obs_name}')
        plt.show()

    # Train the model classically (generation of expectation values classically) and with LASSO regression

    coefficients, L1_norm_coef, optimal_alpha, train_mse, test_mse, train_r2, test_r2 = functions.lasso_regression(X, y)
    print(f'Coefficients (omega)= {coefficients}')
    print(f'L1 norm of omega = {L1_norm_coef}')
    print(f"Optimal alpha: {optimal_alpha}")
    print(f'MSE: training -> {train_mse}; test -> {test_mse}')
    print(f'R^2: training -> {train_r2}; test -> {test_r2}')



