'''
Author: Cesar Hernando

In this file, we train a Machine Learning (ML) model to learn to predict properties of a 2D antiferromagnetic system.

'''

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
import functions
from SA_VQE import SA_VQE_expec_val

mode = 'training'

if mode == 'basic_test':
    # Set the number of qubits in each row/column of the square grid
    dim_grid = (2,2)
    num_qubits = dim_grid[0]*dim_grid[1]

    # Generate the coupling coefficients
    seed = 42
    J_right, J_down = functions.generate_couplings(dim_grid, seed)

    # Visualize the lattice with the randomly generated couplings
    visualize = False
    if visualize:
        functions.visualize_couplings(dim_grid, J_right, J_down)

    # Obtain the Hamiltonian (qml.Hamiltonian) of the 2D Antiferromagnetic lattice
    hamiltonian = functions.hamiltonian(dim_grid, J_right, J_down)

    # Define the observable by its name and qubits it affects
    # Options: obs_name = 'correlation', qubits = (q0, q1)
    obs_name = 'correlation'

    correlations_matrix_diag = np.zeros((num_qubits, num_qubits))
    correlations_matrix_SA_VQE = np.zeros((num_qubits, num_qubits))

    qubits = [(i,j) for i in range(dim_grid[0]) for j in range(dim_grid[1])]

    for i, q0 in enumerate(qubits):
        for j, q1 in enumerate(qubits[i+1:], start=i+1):

    
            obs = functions.observable(dim_grid, obs_name, (q0, q1))

            # Obtain ground state property by diagonalization
            obs_diag, ground_state_energy_diag = functions.ground_state_expectation_value(dim_grid, hamiltonian, obs)
            correlations_matrix_diag[i, j] = obs_diag
            correlations_matrix_diag[j, i] = obs_diag
            #print(f"Expectation value of {obs_name} [Diagonalization]= {obs_diag}")
            #print(f'Ground state energy [Diagonalization] = {ground_state_energy_diag}')
            
            # Obtain ground state property by SA VQE
            depth = 3
            opt_steps = 500
            learning_rate = 0.01
            obs_SA_VQE, ground_state_energy_VQE = SA_VQE_expec_val(dim_grid, hamiltonian, obs, depth, opt_steps, learning_rate)
            correlations_matrix_SA_VQE[i, j] = obs_SA_VQE
            correlations_matrix_SA_VQE[j, i] = obs_SA_VQE
            #print(f"Expectation value of {obs_name} [VQE]= {obs_SA_VQE}")
            #print(f'Ground state energy [VQE] = {ground_state_energy_VQE}')

    
    # Plot the heatmaps
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    im1 = axs[0].imshow(correlations_matrix_diag, cmap='viridis', vmin=np.min(correlations_matrix_diag), vmax=np.max(correlations_matrix_diag))
    axs[0].set_title("Correlation (Diagonalization)")
    plt.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(correlations_matrix_SA_VQE, cmap='viridis', vmin=np.min(correlations_matrix_SA_VQE), vmax=np.max(correlations_matrix_SA_VQE))
    axs[1].set_title("Correlation (SA VQE)")
    plt.colorbar(im2, ax=axs[1])

    for ax in axs:
        ax.set_xlabel("Qubit index")
        ax.set_ylabel("Qubit index")
        ax.set_xticks(range(num_qubits))
        ax.set_yticks(range(num_qubits))

    plt.tight_layout()
    plt.show()
  

elif mode == 'training':
    # General parameters
    dim_grid = (2,2)
    obs_name = 'correlation'
    qubits = ((0,0), (0,1))

    # Training parameters
    num_examples = 100
    mode = 'fourier'

    # VQE parameters
    depth = 3
    opt_steps = 500
    learning_rate = 0.01
    
    # Random fourier map parameters
    delta = 1
    gamma = 0.6
    R = 10
    
    #X, PhiX_quantum, _, y_quantum = functions.generate_training_set(dim_grid, num_examples, obs_name, qubits, mode='quantum', depth=depth, opt_steps=opt_steps, learning_rate=learning_rate)
    X, _, PhiX_fourier, y_classical = functions.generate_training_set(dim_grid, num_examples, obs_name, qubits, mode='fourier', delta=delta, gamma = gamma, R=R)
    print("\nData set generated")

    
    # Train the model with neural network
    print('\nNeural Network\n')
    functions.neural_network(X, y_classical)
    

    # Train the model classically (generation of expectation values classically) and with LASSO regression
    # Without feature map
    coefficients, L1_norm_coef, optimal_alpha, train_mse, test_mse, train_r2, test_r2 = functions.lasso_regression(X, y_classical)
    print('\nLASSO without feature map')
    print(f'Coefficients (omega)= {coefficients}')
    print(f'L1 norm of omega = {L1_norm_coef}')
    print(f"Optimal alpha: {optimal_alpha}")
    print(f'MSE: training -> {train_mse}; test -> {test_mse}')
    print(f'R^2: training -> {train_r2}; test -> {test_r2}')
    '''
    # With quantum feature map
    coefficients, L1_norm_coef, optimal_alpha, train_mse, test_mse, train_r2, test_r2 = functions.lasso_regression(PhiX_quantum, y_quantum)
    print('\nLASSO with quantum feature map')
    print(f'Coefficients (omega)= {coefficients}')
    print(f'L1 norm of omega = {L1_norm_coef}')
    print(f"Optimal alpha: {optimal_alpha}")
    print(f'MSE: training -> {train_mse}; test -> {test_mse}')
    print(f'R^2: training -> {train_r2}; test -> {test_r2}')
    '''
     # With random fourier map
    coefficients, L1_norm_coef, optimal_alpha, train_mse, test_mse, train_r2, test_r2 = functions.lasso_regression(PhiX_fourier, y_classical)
    print('\nLASSO with random Fourier feature map')
    print(f'Coefficients (omega)= {coefficients}')
    print(f'L1 norm of omega = {L1_norm_coef}')
    print(f"Optimal alpha: {optimal_alpha}")
    print(f'MSE: training -> {train_mse}; test -> {test_mse}')
    print(f'R^2: training -> {train_r2}; test -> {test_r2}')
   
    



