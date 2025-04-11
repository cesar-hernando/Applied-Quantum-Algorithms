'''
Author: Cesar Hernando

In this file, we train a Machine Learning (ML) model to learn to predict properties of a 2D antiferromagnetic system.

'''

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
import classes

# Set the number of qubits in each row/column of the square grid
n = 2

# Create an object of the Antiferromagnetic_2D class and compute the Hamiltonian of the system
antiferromagnetic_2D = classes.Antiferromagnetic_2D(n)
H = antiferromagnetic_2D.hamiltonian()
H_matrix = qml.matrix(H)

# Diagonalize the Hamiltonian to find the ground state
eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
ground_state_energy = eigenvalues[0]
ground_state = eigenvectors[:,0]
np.set_printoptions(precision=1, suppress=True)
print(f"Ground state =  {ground_state.real}")
print(f"GS energy = {ground_state_energy}")

# Define the observable to measure: the correlation function between qubits 0 and 1
coefs = (1/3)*np.ones(3)
ops = [qml.PauliX(0) @ qml.PauliX(1), qml.PauliY(0) @ qml.PauliY(1), qml.PauliZ(0) @ qml.PauliZ(1)]
correlation_01 = qml.Hamiltonian(coefs, ops)
correlation_01_matrix = qml.matrix(correlation_01, wire_order=[i for i in range(n*n)])

# Calculate the expectation value of correlation_01 in the ground state
exp_val_corr_01 = (ground_state.conj().T @ correlation_01_matrix @ ground_state).real
print(f"Expectation value of the correlation of qubits 0 and 1 = {exp_val_corr_01}")








