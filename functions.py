'''
In this file, I will define the necessary subroutines for the main file
'''

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import pennylane as qml
import networkx as nx
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError


def generate_couplings(dim_grid, seed=None):
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

    if seed is not None:
        np.random.seed(seed)
    

    J_right = np.random.uniform(-1,1, size=(dim_grid[0], dim_grid[1]-1))
    J_down = np.random.uniform(-1,1, size=(dim_grid[0]-1, dim_grid[1]))

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
            #ops.append(qml.PauliZ(index1)@qml.PauliZ(index2))

    # Fill the vertical interactions
    for i in range(dim_grid[0]-1):
        for j in range(dim_grid[1]):
            index1 = str((i,j))
            index2 = str((i+1,j))
            coef.append(J_down[i,j])
            ops.append(qml.PauliX(index1)@qml.PauliX(index2) + qml.PauliY(index1)@qml.PauliY(index2) + qml.PauliZ(index1)@qml.PauliZ(index2))
            #ops.append(qml.PauliZ(index1)@qml.PauliZ(index2))

    hamiltonian = qml.Hamiltonian(coef, ops)
    

    return hamiltonian

def observable(dim_grid, obs_name, qubits):
    '''
    Computes the Hamiltonian of a 2D antiferromagnetic system of N spins

    Inputs
    ------
    dim_grid: tuple
        Contains the number of rows and columns of the rectangular spin lattice

    obs_name: str
        Name/identification of the observable to measure

    qubits: list
        List containing the tuples representing the x and y coordinates of the qubits in the lattice that we want to measure


    Output
    ------
    observable: qml.Hamiltonian
        Representation of the observable that we want to measure
        
    '''

    n_qubits = dim_grid[0]*dim_grid[1]

    if obs_name == 'correlation':
        # Define the observable to measure: the correlation function between qubits 0 and 1
        coefs = [1/3]
        i0 = str(qubits[0])
        i1 = str(qubits[1])
        ops = [qml.PauliX(i0) @ qml.PauliX(i1) + qml.PauliY(i0) @ qml.PauliY(i1) + qml.PauliZ(i0) @ qml.PauliZ(i1)]
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

    # Calculate the expectation value of correlation_01 in the ground state
    expectation_value = (ground_state.conj().T @ obs_matrix @ ground_state).real

    return expectation_value, ground_state_energy

def generate_training_set(dim_grid, num_examples, observable_name, qubits):
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
    
    obs = observable(dim_grid, observable_name, qubits)
    J_len = int(2*dim_grid[0]*dim_grid[1] - dim_grid[0] - dim_grid[1])  # Number of couplings = len(J_right) + len(J_down)
    X = np.zeros((num_examples, J_len))
    Y = np.zeros(num_examples)
    for l in range(num_examples):
        J_right, J_down = generate_couplings(dim_grid)
        J_right_flat = J_right.flatten()
        J_down_flat = J_down.flatten()
        X[l,:] = np.concatenate((J_right_flat, J_down_flat))
        Y[l], _ = ground_state_expectation_value(dim_grid, hamiltonian(dim_grid, J_right, J_down), obs)

    return X, Y


def lasso_regression(X, y):
    # Split the dataset in training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define an object of the class Lasso and train the model
    lasso = LassoCV(alphas=[1e-4,1e-3,1e-2,1e-1,1], cv=5, random_state=42)
    lasso.fit(X_train, y_train)
    optimal_alpha = lasso.alpha_
    coefficients = lasso.coef_
    L1_norm_coef = np.linalg.norm(coefficients, ord=1)

    # Calculate the predictions for the test set
    y_pred_train = lasso.predict(X_train)
    y_pred_test = lasso.predict(X_test)

    # Evaluate the error and R^2 score in training data and test
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    return coefficients, L1_norm_coef, optimal_alpha, train_mse, test_mse, train_r2, test_r2


def neural_network(X,y):
    # Split the dataset in training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dim_input = X_test.shape[1]
    num_examples = X_test.shape[0]
    # Define the model
    seed = 42
    tf.random.set_seed(seed)
    
    model = tf.keras.Sequential(([
        tf.keras.layers.Input(dim_input),
        tf.keras.layers.Dense(dim_input, activation="relu"),
        tf.keras.layers.Dense(dim_input/2, activation="relu"),
        tf.keras.layers.Dense(dim_input/4, activation="relu"),
        tf.keras.layers.Dense(1, activation=None)
        ]))
    
    # Set the optimizer and loss function
    opt = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(optimizer=opt, loss=MeanSquaredError())
    # Set an early stopping criteria using validation loss
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, min_delta=0.001)
    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=int(num_examples/5), callbacks=[early_stop])
    # Compute the output of the model on the test data
    output = model.predict(X_test)
    # Calculate and print the accuracy
    print(f"r2 score on test data = {r2_score(output, y_test)}")
    print(f"Mean squared error on test data = {mean_squared_error(output, y_test)}")
    