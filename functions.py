'''
In this file, I will define the necessary subroutines for the main file
'''

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import pennylane as qml
import networkx as nx


def generate_couplings(dim_grid, rng: Optional[np.random.RandomState] = None):
    '''
    Generate the horizontal and vertical couplings between the qubits in the square lattice sampling from a uniform distribution {-1,1}

    Inputs
    ------
    dim_grid: tuple
        Contains the number of rows and columns of the rectangular spin lattice

    rng: numpy.random.mtrand.RandomState
        Random number generator (for reproducible results)

    Outputs
    -------
    J_right: np.ndarray
        Horizontal couplings
        Dimension dim_grid[0] x (dim_grid[1]-1)

    J_down: np.ndarray
        Vertical couplings
        Dimension (dim_grid[0]-1) x dim_grid[1]
    '''

    if rng is None:
        rng = np.random.RandomState(seed=42)
    
    J_right = np.array([[rng.choice([+1,-1]) for j in range(dim_grid[1]-1)] for i in range(dim_grid[0])])
    J_down = np.array([[rng.choice([+1,-1]) for j in range(dim_grid[1])] for i in range(dim_grid[0]-1)])

    return J_right, J_down

def visualize_couplings(dim_grid, J_right, J_down):
    '''
    Generates a plot to visualize the value of the couplings between spins

    Inputs
    -------
    dim_grid: tuple
        Contains the number of rows and columns of the rectangular spin lattice

    J_right: np.ndarray
        Horizontal couplings
        Dimension dim_grid[0] x (dim_grid[1]-1)

    J_down: np.ndarray
        Vertical couplings
        Dimension (dim_grid[0]-1) x dim_grid[1]
    '''
    rows, columns = dim_grid
    G = nx.Graph()
    pos = {}

    # Add nodes and positions
    for i in range(rows):
        for j in range(columns):
            qubit = (i,j)
            G.add_node(qubit)
            pos[qubit] = (j,-i)

    # Add horizontal edges from J_right
    for i in range(rows):
        for j in range(columns-1):
            q1 = (i,j)
            q2 = (i,j+1)
            sign = J_right[i,j]
            G.add_edge(q1, q2, color='red' if sign==1 else 'blue', weight=sign)

    # Ad vertical edges from J_down
    for i in range(rows - 1):
        for j in range(columns):
            q1 = (i, j)
            q2 = (i + 1, j)
            sign = J_down[i, j]
            G.add_edge(q1, q2, color='red' if sign == 1 else 'blue', weight=sign)

    # Draw the network
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, node_color='lightgray', edge_color=edge_colors, node_size=700, width=2)
    plt.title("2D Spin Grid with Coupling Signs\nRed: +1 (ferromagnetic), Blue: −1 (antiferromagnetic)")
    plt.axis('off')
    plt.show()



def hamiltonian(dim_grid, J_right, J_down):
    '''
    Computes the Hamiltonian of a 2D antiferromagnetic system of N spins

    Inputs
    ------
    dim_grid: tuple
        Contains the number of rows and columns of the rectangular spin lattice

     J_right: np.ndarray
        Horizontal couplings
        Dimension n x (n-1)

    J_down: np.ndarray
        Vertical couplings
        Dimension (n-1) x n
    
    Output
    ------
    hamiltonian: qml.Hamiltonian
        Hamiltonian representation of the system for specific coupling coefficients
        
    '''

    coef = []
    ops = []

    # Fill the horizontal interactions
    for i in range(dim_grid[0]):
        for j in range(dim_grid[1]-1):
            index1 = str((i,j))
            index2 = str((i,j+1))
            coef.append(J_right[i,j])
            ops.append(qml.PauliX(index1)@qml.PauliX(index2) + qml.PauliY(index1)@qml.PauliY(index2) + qml.PauliZ(index1)@qml.PauliZ(index2))

    # Fill the vertical interactions
    for i in range(dim_grid[0]-1):
        for j in range(dim_grid[1]):
            index1 = str((i,j))
            index2 = str((i+1,j))
            coef.append(J_down[i,j])
            ops.append(qml.PauliX(index1)@qml.PauliX(index2) + qml.PauliY(index1)@qml.PauliY(index2) + qml.PauliZ(index1)@qml.PauliZ(index2))

    hamiltonian = qml.Hamiltonian(coef, ops)

    return hamiltonian

def observable(dim_grid, obs_name):
    '''
    Computes the Hamiltonian of a 2D antiferromagnetic system of N spins

    Inputs
    ------
    dim_grid: tuple
        Contains the number of rows and columns of the rectangular spin lattice

    obs_name: str
        Name/identification of the observable to measure

    Output
    ------
    observable: qml.Hamiltonian
        Representation of the observable that we want to measure
        
    '''

    n_qubits = dim_grid[0]*dim_grid[1]

    if obs_name == 'corr01':
        # Define the observable to measure: the correlation function between qubits 0 and 1
        coefs = (1/3)*np.ones(3)
        i0 = str((0,0))
        i1 = str((0,1))
        ops = [qml.PauliX(i0) @ qml.PauliX(i1), qml.PauliY(i0) @ qml.PauliY(i1), qml.PauliZ(i0) @ qml.PauliZ(i1)]
        observable = qml.Hamiltonian(coefs, ops)

    elif obs_name == 'magnetization':
        # Define the observable to measure: magnetization
        coefs = np.ones(n_qubits)
        ops = [qml.PauliZ(str((x,y))) for x in range(dim_grid[0]) for y in range(dim_grid[1])]
        observable = qml.Hamiltonian(coefs, ops)
    
    return observable


def ground_state_expectation_value(dim_grid, hamiltonian, observable):
    '''
    Computes the ground state expectation value of a given observable classically

    Inputs
    ------
    dim_grid: tuple
        Contains the number of rows and columns of the rectangular spin lattice

    hamiltonian: qml.Hamiltonian
        Hamiltonian representation of the system for specific coupling coefficients

    observable: qml.Hamiltonian
        Representation of the k-local observable whose ground state expectation value we want to evaluate

    Output
    ------
    expectation_value: float
        Formula: Tr(\rho(x) O)
        
    '''
     
    # Transform the qml.Hamiltonian objects to np matrices
    wires = [str((i, j)) for i in range(dim_grid[0]) for j in range(dim_grid[1])]
    H_matrix = qml.matrix(hamiltonian, wire_order=wires)
    obs_matrix = qml.matrix(observable, wire_order=wires)

    # Diagonalize the Hamiltonian to find the ground state
    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
    ground_state_energy = eigenvalues[0]
    ground_state = eigenvectors[:,0]

    # Print the ground state and its energy
    verbose = False
    if verbose:
        np.set_printoptions(precision=1, suppress=True)
        print(f"Ground state =  {ground_state.real}")
        print(f"GS energy = {ground_state_energy}")

    # Calculate the expectation value of correlation_01 in the ground state
    expectation_value = (ground_state.conj().T @ obs_matrix @ ground_state).real

    return expectation_value, ground_state_energy

def generate_training_set(dim_grid, num_examples, observable_name):
    '''
    Generate the training set for the model

    Inputs
    ------
    dim_grid: tuple
        Contains the number of rows and columns of the rectangular spin lattice

    num_examples: int
       Number of training examples |{x^l={J_{ij}^l, y^l}|

    observable_name: str
        Name/identification of the k-local observable whose ground state expectation value we want to evaluate

    Output
    ------
    X: np.ndarray
        Matrix of dimension num_examples x J_len containing num_examples of the input vector
    
    Y: np.ndarray
        Vector of length num_examples with the expectation values of the measured observable
        
    ''' 

    rng = np.random.RandomState(seed=42)
    obs = observable(observable_name)
    J_len = int(2*dim_grid[0]*dim_grid[1] - dim_grid[0] - dim_grid[1])  # Number of couplings = len(J_right) + len(J_down)
    X = np.zeros((num_examples, J_len))
    Y = np.zeros(num_examples)
    for l in range(num_examples):
        J_right, J_down = generate_couplings(dim_grid, rng)
        J_right_flat = J_right.flatten()
        J_down_flat = J_down.flatten()
        X[l,:] = np.concatenate((J_right_flat, J_down_flat))
        Y[l] = ground_state_expectation_value(dim_grid, hamiltonian(dim_grid, J_right, J_down), obs)

    return X, Y




