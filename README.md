# NeuralNetwork-FLRW
A neural network approach for the FLRW universe in Euclidean Regge calculus




1 OneStep_NN.py is the "library" for the neural network project.
  It should be in the current directory to easily import it!

2 Its documentation can be accessed using the accompanying Jupyter notebook Playground.ipynb.
  Otherwise, its documentation can be accessed in Python using the following statements:
        import OneStep_NN as osnn
        osnn.documentation('help')
  This shows a list of commands of the implemented functions.
  Replace 'help' by the respective command to read its documentation.

3 The documentation contains four sample codes that can be copied and used immediately.

4 The project has been implemented using:
        -> Python 3.9
        -> Tensorflow 2.10.1, as well as the version of Keras coming with it (2.10.0)
        -> Jupyter Notebook 6.4.3




The neural network is capable of solving the Euclidean Regge equations of motion for a single time step. If no solution can be found right away, try:
        -> to increase the learning rate
        -> to decrease the clipnorm
        -> to switch to the other solver
before altering more advanced settings, which can be found in the documentation of the respective functions.