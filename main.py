'''
Author: Cesar Hernando

In this file, we train a Machine Learning (ML) model to learn to predict properties of a 2D antiferromagnetic system.

'''

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
import classes

# Compute the Hamiltonian of the 2D antiferromagnetic system of 9 qubits

n = 2
antiferromagnetic_2D = classes.Antiferromagnetic_2D(n)
H = antiferromagnetic_2D.hamiltonian()
H_mat = qml.matrix(H)
print(H_mat.shape)





