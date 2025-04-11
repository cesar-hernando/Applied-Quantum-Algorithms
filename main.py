'''
Author: Cesar Hernando

In this file, we train a Machine Learning (ML) model to learn to predict properties of a 2D antiferromagnetic system.

'''

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
import functions

# Compute the Hamiltonian of the 2D antiferromagnetic system of 9 qubits

n = 2
qubit_coord = [(i,j) for i in range(n) for j in range(n)]
J_right, J_down = functions.generate_couplings(n)
H = functions.hamiltonian(n, qubit_coord, J_right, J_down)
H_matrix = qml.matrix(H)





