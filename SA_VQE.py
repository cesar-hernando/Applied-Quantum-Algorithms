'''
In this file, I estimate the expectation value of a certain observable in the ground state of the Hamiltonian
using Symmetry-Adapted VQE
'''

import pennylane.numpy as np
import pennylane as qml


def SA_VQE_expec_val(dim_grid, hamiltonian, observable, depth, opt_steps, learning_rate):
    '''
    Inputs
    ------
    dim_grid: tuple
        Contains the number of rows and columns of the rectangular spin lattice

    hamiltonian: qml.Hamiltonian
        Hamiltonian representation of the system for specific coupling coefficients

    observable: qml.Hamiltonian
        Representation of the k-local observable whose ground state expectation value we want to evaluate

    depth: int
        Number of times that we apply the RZ,RX,RZ,CNOT block in VQE

    opt_steps: int
        Number of iterations in the optimization

    learning_rate: float
        Adam initial step size

    Output
    ------
    obs_ground_state: float
        Estimate of the expectation value of the observable in the ground state

    ground_state_energy: float
        Estimate of the expectation value of the Hamiltonian in the ground state
    '''

    n_qubits = dim_grid[0]*dim_grid[1]
    qubit_indices = [str((i,j)) for i in range(dim_grid[0]) for j in range(dim_grid[1])]
    dev = qml.device("default.qubit", wires=qubit_indices)
    n_params = n_qubits*3*depth

    def hea_ansatz(params):
        for d in range(depth):
            for q in range(n_qubits):
                x = q // dim_grid[1]
                y = q % dim_grid[1]
                wire = str((x,y))
                qml.RZ(params[3*d, q], wires=wire)
                qml.RX(params[3*d+1, q], wires=wire)
                qml.RZ(params[3*d+2, q], wires=wire)
                
            
            for i in range(0, n_qubits-1, 2):
                x_ctrl = i // dim_grid[1]
                y_ctrl = i % dim_grid[1]
                wire_ctrl = str((x_ctrl,y_ctrl))
                x_target = (i+1) // dim_grid[1]
                y_target = (i+1) % dim_grid[1]
                wire_target = str((x_target, y_target))
                qml.CNOT(wires=[wire_ctrl, wire_target])


    @qml.qnode(dev)
    def energy(params):
        hea_ansatz(params)
        return qml.expval(hamiltonian)
    
    @qml.qnode(dev)
    def observable_exp_val(params):
        hea_ansatz(params)
        return qml.expval(observable)


    def optimizer(params):
        '''
        Optimizer to adjust the parameters.
        Args:
            params (np.array): An array with the trainable parameters of the QAOA ansatz.
        
        Returns:
            (np.array): An array with the optimized parameters of the HEA ansatz.
        ''' 
    
        opt = qml.AdamOptimizer(learning_rate)

        for _ in range(opt_steps):
            params = opt.step(energy, params)
        
        return params


    params_initial = np.random.uniform(-np.pi, np.pi, size=(3*depth, n_qubits), requires_grad=True)
    params_final = optimizer(params_initial)
    ground_state_energy = energy(params_final)
    obs_ground_state = observable_exp_val(params_final)

    return obs_ground_state.item(), ground_state_energy.item()




