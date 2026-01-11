# FLRW-Net
A neural network approach for the FLRW universe in Euclidean Regge calculus


## Get started
To set up FLRW-Net on your local device, do the following:
1. Clone the repository from GitHub.
1. Go to the project root 'NEURALNETWORK-FLRW' in your preferred editor.
1. Run `./setup_flrw_net.sh` from project root
1. Congrats. Now you can start FLRW-Net from project root with: `pdm start-flrw-net`

## Note
The neural network is capable of solving the Euclidean Regge equations of motion for a single time step. If no solution can be found right away, try:
        -> to increase the learning rate
        -> to decrease the clipnorm
        -> to switch to the other solver
before altering more advanced settings, which can be found in the documentation of the respective functions.
