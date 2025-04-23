'''
Author: Cesar Hernando

In this file, we train a Machine Learning (ML) model to learn to predict properties of a 2D antiferromagnetic system.

'''

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
import functions

mode = 'training'

if mode == 'basic_test':
    # Set the number of qubits in each row/column of the square grid
    n = 2

    # Generate the coupling coefficients
    J_right, J_down = functions.generate_couplings(n)

    # Obtain the Hamiltonian (qml.Hamiltonian) of the 2D Antiferromagnetic lattice
    H = functions.hamiltonian(n, J_right, J_down)

    # Define the observable
    correlation_01 = functions.observable('corr01')

    # Calculate the ground state expectation value of the observable
    exp_val_corr_01 = functions.ground_state_expectation_value(n, H, correlation_01)
    print(f"Expectation value of the correlation of qubits 0 and 1 = {exp_val_corr_01}")

elif mode == 'training':
    n = 2
    num_examples = 1000
    obs_name = 'corr01'

    X, Y = functions.generate_training_set(n, num_examples, obs_name)
    plt.figure()
    plt.plot(range(len(Y)), Y)
    plt.xlabel('Coupling random realization')
    plt.ylabel(f'Expectarion value of {obs_name}')
    plt.show()








