'''
In this file, I define a class to predict properties of a 2D antiferromagnetic system using a ML model
'''

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

class Antiferromagnetic_2D:
    def __init__(self, n:int):
        self.n = n
    
    def generate_couplings(self):
        '''
        Generate the horizontal and verticals between the qubits in the square lattice

        Inputs
        ------
        n: int
            Number of qubits in each row/column
        
        Outputs
        -------
        J_right: np.ndarray
            Horizontal couplings
            Dimension n x (n-1)

        J_down: np.ndarray
            Vertical couplings
            Dimension (n-1) x n
        '''
        
        rng = np.random.RandomState(seed=42)
        J_right = np.array([[rng.choice([+1,-1]) for j in range(self.n - 1)] for i in range(self.n)])
        J_down = np.array([[rng.choice([+1,-1]) for j in range(self.n)] for i in range(self.n - 1)])

        return J_right, J_down
    
    def hamiltonian(self):
        '''
        Computes the Hamiltonian of a 2D antiferromagnetic system of N spins

        Inputs
        ------
        n: int
            Number of qubits in each row/column (n = sqrt(N))

        qubit_coord: list
            List with the coordinates (tuples) of all the qubits in the square lattice
            Dimension n x n
            
        J_right: np.ndarray
            Matrix with the horizontal couplings
            Dimension (n-1) x n

        J_down: np.ndarray
            Matrix with the vertical couplings
            Dimension n x (n-1)

        Output
        ------
        hamiltonian: np.ndarray
            Matrix of the Hamiltonian in the computational basis
            Dimension 2**N x 2**N
            
        '''

        J_right, J_down = self.generate_couplings()

        coef = []
        ops = []

        # Fill the horizontal interactions
        for i in range(self.n):
            for j in range(self.n - 1):
                index1 = str((i,j))
                index2 = str((i,j+1))
                coef.append(J_right[i,j])
                ops.append(qml.PauliX(index1)@qml.PauliX(index2) + qml.PauliY(index1)@qml.PauliY(index2) + qml.PauliZ(index1)@qml.PauliZ(index2))

        # Fill the vertical interactions
        for i in range(self.n - 1):
            for j in range(self.n):
                index1 = str((i,j))
                index2 = str((i+1,j))
                coef.append(J_down[i,j])
                ops.append(qml.PauliX(index1)@qml.PauliX(index2) + qml.PauliY(index1)@qml.PauliY(index2) + qml.PauliZ(index1)@qml.PauliZ(index2))

        hamiltonian = qml.Hamiltonian(coef, ops)

        return hamiltonian
