# Applied-Quantum-Algorithms
 Predicting unknown observables on ground states

 In this project I will use different supervised learning methods to solve a quantum many body physics problem.
 Specifically, the task is to learn from data to predict ground state expectation values of unknown
 observables for a class of local Hamiltonians. For this, I will compare different methods:
 a quantum LASSO regression model that involves executing VQE; a LASSO regression that involves using a random Fourier feature map; and a Deep Neural Network.

 In order to train these models and compare their performance, you should set the variable "mode" in main.py to 'B'. If you want to check how
 well the estimated ground state energies and observables match with the exact ones obtained by diagonalization, set "mode" to 'A'. Furthermore, different 
 parameters such as the lattice size, number of examples and hyperparameters of different models can be changed in the parameters.py file.

 The file SA_VQE.py include the functions related to the VQE algorithm. IMPORTANT: in the end, I did not use the symmetry-adapted VQE but the hardware-efficient VQE, but I did not change the file name. I hope this comment is enough to avoid confusion.
 
 The fourier_feature_mappy file contains the functions related to the classical
 LASSO model proposed in Ref. 3. Finally, the functions.py contains the rest of the functions to generate couplings, define hamiltonian and observble,
 calculate the ground state using numpy diagonalization, generating the training dataset, and defining the ML models.

 The required libraries to run the code are: numpy, matplotlib, pennylane, networkx, sklearn, and tensorflow.
