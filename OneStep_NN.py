# Import numpy and tensorflow libraries
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Set standard precision from float32 to float64
tf.keras.backend.set_floatx('float64')

def doc_consistency_check():
    print("---------- Documentation of: consistency_check ----------")
    print("Check whether a solution exists for the given boundary edges and cosmological constant.")
    print("Increase the number of printed decimals by altering the kwarg 'decimals'. It defaults to 4.")
    print("Usage: osnn.consistency_check(N1, N3, lamb, l1, l2, decimals=4)")
    print("Returns: Bool")
    print("----------------------------------------------------------")
    print()
    
def doc_lambda_limit():
    print("---------- Documentation of: lambda_limit ----------")
    print("Computes the largest possible value of lambda for which there is a solution.")
    print("Usage: osnn.lambda_limit(N1, N3, l1, l2)")
    print("Returns: float")
    print("-----------------------------------------------------")
    print()
    
def doc_oom():
    print("---------- Documentation of: oom ----------")
    print("Compute the order of magnitude of a given number.")
    print("Usage: osnn.oom(number)")
    print("Returns: integer")
    print("--------------------------------------------")
    print()
    
def doc_NN():
    print("---------- Documentation of: NN ----------")
    print("Neural network to determine solutions for 'sufficiently small' values of lambda.")
    print("'sufficiently small' are values smaller than approximately 0.96 lambda_limit.")
    print("Default arguments: N1=10, N3=5, lamb=1e-2, **kwargs")
    print()
    print("Built-in functions:")
    print("get_params(): returns the parameters handed to the model.")
    print("set_weights(weights): overwrites current trainable weights with custom ones.")
    print("get_weights(): returns the current trainable weights.")
    print("training(inputs, epochs): training without initialising the neural network again.")
    print("compute_loss(array): computes the network's loss as the squared equations of motion.")
    print()
    print("Usages:")
    print("model = osnn.NN(**params)")
    print("""adam_optimizer = osnn.tf.keras.optimizers.Adam(
        learning_rate=lr,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        decay=0.9,
        clipnorm=cv,
        clipvalue=None,
        global_clipnorm=None)""")
    print("model.compile(optimizer=adam_optimizer)")
    print("model.get_params() ---------- returns: None")
    print("model.set_weights([[w1],[w2],[w3]])")
    print("model.get_weights() ---------- returns: array")
    print("X=tf.constant([[l1, a, l2]],dtype=tf.float64); model(X) returns the model's prediction Y=[[l1, m1, l2]]")
    print("model.training(X, epochs) ---------- returns: argmin, minimum_loss, (loss_array only if rl=True)")
    print("model.compute_loss([[l1, m1, l2]]) or model.compute_loss(model(X)) --- returns: float")
    print("-------------------------------------------")
    print()
    
def doc_NN2():
    print("---------- Documentation of: NN2 ----------")
    print("Neural network to determine solutions for 'sufficiently large' values of lambda.")
    print("They partly overlap with the values to which NN can be effectively applied.")
    print("Usage: model = osnn.NN2(**params)")
    print("Otherwise analogous to 'NN'.")
    print("--------------------------------------------")
    print()
    
def doc_OneStepSolver():
    print("---------- Documentation of: OneStepSolver ----------")
    print("Solving routine for one time step making use of NN, i.e., for sufficiently small values of lambda.")
    print("""The basic kwargs:
         N1       : number of equilateral edges in the three-sphere
         N3       : number of equilateral tetrahedra in the three-sphere
         lamb     : cosmological constant
         epochs   : number of epochs for training
         lr       : learning rate
         rtw      : if True, returns the trained weights
         graph    : if True, returns the training graph
         weights  : trainable weights set to the hidden layer""")
    print()
    print("""The advanced kwargs:
         repetition: maximum number of repeating one calculation with different learning rates
         lr_mult  : multiplier for the learning rate
         lr_add   : adding to learning rate
         lr_limit : limit for switching between multiplication and addition
         loss_accuracy: limit to consider some parameters a solution of the model
         epochs_mult: factor by which the current number of epochs is increased upon addition of lr_add""")
    print()
    print("""The adam parameter kwargs: --- only adjust if you know what you are doing! ---
         adam_beta_1=0.9
         adam_beta_2=0.999
         adam_epsilon=1e-07
         adam_amsgrad=False
         adam_decay=0.9
         adam_clipnorm=1.              # clip value (gradient cut-off norm) of the optimizer
         adam_clipvalue=None           # clip value (gradient cut-off) of the optimizer
         adam_global_clipnorm=None""")
    print()
    print("Usage:")
    print("""training_kwargs = {
        'N1': 10,
        'N3': 5,
        'lamb': 17,
        'epochs' : 5000,
        'lr' : 1e-2,
        'rtw': True,
        'graph':True,
        'adam_clipnorm' : 1.
    }""")
    print("X=tf.constant([[l1, a, l2]],dtype=tf.float64) e.g., X=tf.constant([[1, 0.025, 2]],dtype=tf.float64)")
    print("output, trained_weights, losses = osnn.OneStepSolver(inputs=X, **training_kwargs)")
    print("Returns: output, trained_weights, losses (if rtw=True, else: output, losses)")
    print("------------------------------------------------------")
    print()
    
def doc_OneStepSolver2():
    print("---------- Documentation of: OneStepSolver2 ----------")
    print("Solving routine for one time step making use of NN2, i.e., for sufficiently large values of lambda.")
    print("As a starting point, initialize 'lr' as 1e-1, and 'adam_clipnorm' as 1e-2 together with l1=1, l2=2.")
    print("Input form: X2 = tf.constant([[l1, np.arctan(np.sqrt(a)*np.abs(l2-l1)), l2]],dtype=tf.float64)")
    print("with a being 2/5, 1/2, and (3+sqrt(5))/2 for the 5-cell, the 16-cell and the 600-cell, respectively.")
    print("""The basic kwargs:
         N1       : number of equilateral edges in the three-sphere
         N3       : number of equilateral tetrahedra in the three-sphere
         lamb     : cosmological constant
         epochs   : number of epochs for training
         lr       : learning rate
         rtw      : if True, returns the trained weights
         graph    : if True, returns the training graph
         weights  : trainable weights set to the hidden layer""")
    print()
    print("""The adam parameter kwargs: --- only adjust if you know what you are doing! ---
         adam_beta_1=0.9
         adam_beta_2=0.999
         adam_epsilon=1e-07
         adam_amsgrad=False
         adam_decay=0.9
         adam_clipnorm=1.              # clip value (gradient cut-off norm) of the optimizer
         adam_clipvalue=None           # clip value (gradient cut-off) of the optimizer
         adam_global_clipnorm=None""")
    print("Usage:")
    print("""X2 = tf.constant([[1., np.arctan(np.sqrt(2/5)), 2.]],dtype=tf.float64)

training_kwargs2 = {
    'N1': 10,
    'N3': 5,
    'lamb': 10,
    'epochs' : 5000,
    'lr' : 1e-1,
    'graph': False,
    'adam_clipnorm' : 1e-2,
    'rtw': True
}

output, trained_weights, losses = osnn.OneStepSolver2(inputs=X2, **training_kwargs2)
""")
    print("Returns: output, trained_weights, losses (if rtw=True, else: output, losses)")
    print("-------------------------------------------------------")
    print()

def doc_lambda_array():
    print("---------- Documentation of: lambda_array ----------")
    print("Computes the linspaced array of possible values of lambda")
    print("and prepends logspaced array before the linspaced one.")
    print("Kwargs: 'lin': Accuracy of linspacing 10^-lin. Defaults to 1.")
    print("        'log': Includes lambdas up to 'log' orders of magnitude smaller than first linspace point. Defaults to 5.")
    print("Usage: osnn.lambda_array(N1, N3, l1, l2, lin=1, log=5)")
    print("Returns: array")
    print("-----------------------------------------------------")
    print()
    
def doc_choose_solver():
    print("---------- Documentation of: choose_solver ----------")
    print("Function that determines which of the two OneStepSolvers might be more useful.")
    print("Usage: osnn.choose_solver(inputs, N1=10, N3=5, lamb=1e-3, **kwargs)")
    print("Returns: None")
    print("------------------------------------------------------")
    print()
    
def doc_network_architecture():
    print("---------- Documentation of: architecture ----------")
    print("Schematic representation of the neural network's  architecture (NN and NN2).")
    print(r"""Define the neural network. It has the following structure:
     I H O (Input, Hidden, Output)
    -o-o-o-
      \
    -o-o-o-
      /
    -o-o-o-""")
    print("------------------------------------------------------")
    print()

def doc_example_solver1():
    print("---------- Documentation of: Solver 1 ----------")
    print("Below is an example usage of the OneStepSolver.")
    print("""X = tf.constant([[1., 0.025, 2.]],dtype=tf.float64)

# Set keyword arguments for training
training_kwargs = {
    'N1': 10,
    'N3': 5,
    'lamb': 10,
    'epochs' : 5000,
    'lr' : 1e-3,
    'graph': True,
    'adam_clipnorm' : 1.
}

output, losses = osnn.OneStepSolver(inputs=X, **training_kwargs)""")
    print("---------------------------------------------------")
    print()
    
def doc_example_loop_solver1():
    print("---------- Documentation of: Solver 1 ----------")
    print("Below is an example usage of the OneStepSolver in a loop for different lambda.")
    print("""# Define input
X = tf.constant([[1., 0.025, 2.]],dtype=tf.float64)

lambdas = osnn.lambda_array(N1=10, N3=5, l1=X[0,0], l2=X[0,2], lin=0, log=5)
struts = []
loss_data = []

training_kwargs = {
    'N1': 10,
    'N3': 5,
    'epochs' : 10000,     # number of epochs for training
    'lr' : 1e-3,         # learning rate of the optimizer
    'graph': True,
    'adam_clipnorm' : 1., # clip value (gradient cut-off) of the optimizer
    'rtw': True
}

for i in range(len(lambdas)):
    output, trained_weights, losses = osnn.OneStepSolver(inputs=X, lamb=lambdas[i], **training_kwargs)
    struts.append(output[1])
    loss_data.append(losses[-1])

    # Set keyword arguments for training
    training_kwargs = {
        'N1': 10,
        'N3': 5,
        'epochs' : 10000,     # number of epochs for training
        'lr' : 1e-3,         # learning rate of the optimizer
        'graph': False,
        'adam_clipnorm' : 1., # clip value (gradient cut-off) of the optimizer
        'rtw': True,
        'weights': trained_weights
    }""")
    print("---------------------------------------------------")
    print()
    
def doc_example_solver2():
    print("---------- Documentation of: Solver 2 ----------")
    print("Below is an example usage of OneStepSolver2.")
    print("""l1 = 1
l2 = 2
atan_m1 = np.arctan(np.sqrt(2/5)*np.abs(l2-l1))
X2 = tf.constant([[l1, atan_m1, l2]],dtype=tf.float64)

# Set keyword arguments for training
training_kwargs2 = {
    'N1': 10,
    'N3': 5,
    'lamb': 17.583,
    'epochs' : 5000,
    'lr' : 1e-1,       
    'graph': True,
    'adam_clipnorm': 1e-2
}

output2, losses2 = osnn.OneStepSolver2(inputs=X2, **training_kwargs2)""")
    print("---------------------------------------------------")
    print()
    
def doc_example_loop_solver2():
    print("---------- Documentation of: Solver 2 ----------")
    print("Below is an example usage of OneStepSolver2 in a loop for different lambda.")
    print("""# Define input
l1 = 1
l2 = 2
atan_m1 = np.arctan(np.sqrt(2/5)*np.abs(l2-l1))
X2 = tf.constant([[l1, atan_m1, l2]],dtype=tf.float64)

lambdas2 = np.linspace(17,17.583, 20)
struts2 = []
loss_data2 = []

training_kwargs2 = {
    'N1': 10,
    'N3': 5,
    'epochs' : 5000,     # number of epochs for training
    'lr' : 1e-1,         # learning rate of the optimizer
    'graph': True,
    'adam_clipnorm' : 1e-2, # clip value (gradient cut-off) of the optimizer
    'rtw': True
}

for i in range(len(lambdas2)):
    output, trained_weights, losses = osnn.OneStepSolver2(inputs=X2, lamb=lambdas2[i], **training_kwargs2)
    struts2.append(output[1])
    loss_data2.append(losses[-1])

    # Set keyword arguments for training
    training_kwargs2 = {
        'N1': 10,
        'N3': 5,
        'epochs' : 5000,     # number of epochs for training
        'lr' : 1e-1,         # learning rate of the optimizer
        'graph': False,
        'adam_clipnorm' : 1e-2, # clip value (gradient cut-off) of the optimizer
        'rtw': True,
        'weights': trained_weights
    }""")
    print("---------------------------------------------------")
    print()

def copyright():
    print("Copyright 2024 Florian Emanuel Hilpert")
    print("Author: Florian Emanuel Hilpert")
    print()

def documentation(string="help"):
    if string=="help":
        print("---------- OneStep_NN functions: ----------")
        print("'consistency_check'")
        print("'lambda_limit'")
        print("'oom'")
        print("'NN'")
        print("'NN2'")
        print("'OneStepSolver'")
        print("'OneStepSolver2'")
        print("'lambda_array'")
        print("'choose_solver'")
        print("'architecture'")
        print("'all' prints documentation of all functions")
        print()
        print("'Solver1_example'")
        print("'Solver1_loop_example'")
        print("'Solver2_example'")
        print("'Solver2_loop_example'")
        print("-------------------------------------------")
        print()
        print("'copyright'")
        
    elif string=="consistency_check":
        doc_consistency_check()
        
    elif string=="lambda_limit":
        doc_lambda_limit()
        
    elif string=="oom":
        doc_oom()
        
    elif string=="NN":
        doc_NN()
        
    elif string=="NN2":
        doc_NN2()
        
    elif string=="OneStepSolver":
        doc_OneStepSolver()
        
    elif string=="OneStepSolver2":
        doc_OneStepSolver2()
        
    elif string=="lambda_array":
        doc_lambda_array()
        
    elif string=="choose_solver":
        doc_choose_solver()
        
    elif string=="architecture":
        doc_network_architecture()

    elif string=="copyright":
        copyright()
        
    elif string=="all":
        copyright()
        doc_consistency_check()
        doc_lambda_limit()
        doc_oom()
        doc_NN()
        doc_NN2()
        doc_OneStepSolver()
        doc_OneStepSolver2()
        doc_lambda_array()
        doc_choose_solver()
        doc_network_architecture()
        
    elif string=="Solver1_example":
        doc_example_solver1()
    
    elif string=="Solver1_loop_example":
        doc_example_loop_solver1()
        
    elif string=="Solver2_example":
        doc_example_solver2()
        
    elif string=="Solver2_loop_example":
        doc_example_loop_solver2()
    
    else:
        print("I think you misspelled. Please correct your command: ", string)

# Check if EOMm allow for solutions before computations
def consistency_check(N1, N3, lamb, l1, l2, decimals=4):
    if N3 == 600:
        nte = 5
    elif N3 == 16:
        nte = 4
    else:
        nte = 3

    value = lamb * (l1**2 + l2**2) * N3 / N1
    limit = 12 * np.sqrt(2) * (2*np.pi - nte*np.arccos(1/3))
    
    # Format the values using Python's string formatting
    formatted_value = "{:.{}f}".format(value, decimals)
    formatted_limit = "{:.{}f}".format(limit, decimals)
    
    if value >= limit:
        if decimals != 0:
            tf.print(f"There is no solution since value={formatted_value} >= limit={formatted_limit}")
        return False
    else:
        if decimals != 0:
            tf.print(f"There is a solution since value={formatted_value} < limit={formatted_limit}")
        return True

# Compute the largest value of lambda for which there is a solution
def lambda_limit(N1, N3, l1, l2):
    if N3 == 600:
        nte = 5
    elif N3 == 16:
        nte = 4
    else:
        nte = 3

    limit = 12 * np.sqrt(2) * (2*np.pi - nte*np.arccos(1/3))
    value = (l1**2 + l2**2) * N3 / N1
    return (limit / value - 1e-14)

# Compute the order of magnitude of a given number
def oom(number):
    if number == 0:
        return float('-inf')
    else:
        return np.floor(np.log10(np.abs(number)))
    
# Define the hidden layer of the neural network
class HiddenLayer(tf.keras.layers.Layer):
    # Initialize the hidden layer.
    def __init__(self):
        super(HiddenLayer, self).__init__()
        
        # Define the weights for the three neurons respectively ...
        weights_1 = tf.constant([[1.]], dtype=tf.float64)
        weights_2 = tf.constant([[0.], [1.], [0.]], dtype=tf.float64)
        weights_3 = tf.constant([[1.]], dtype=tf.float64)
        
        # ... and the biases 
        biases = tf.constant([0., 0., 0.], dtype=tf.float64)
        
        # Set the weights for the hidden layer:
        # the weights for the boundary spatial edges and the biases are set to be non-trainable
        # whereas the weights for the strut can be learned
        self.weights_1 = tf.Variable(initial_value=weights_1, trainable=False, dtype=tf.float64)
        self.weights_2 = tf.Variable(initial_value=weights_2, trainable=True, dtype=tf.float64)
        self.weights_3 = tf.Variable(initial_value=weights_3, trainable=False, dtype=tf.float64)
        self.biases = tf.Variable(initial_value=biases, trainable=False, dtype=tf.float64)
    
    # Set the trainable weights to the values of a pre-trained model
    def set_weights(self, weights_2):
        self.weights_2.assign(weights_2)
        
    # Define an activation function for the strut-neurons
    def strut_activation(self, element_1, element_2, element_3):
        output = tf.math.sqrt(element_2 + tf.constant(3/8,dtype=tf.float64)) * tf.math.abs(element_1 - element_3)
        return output
    
    # Forward-feed when the hidden layer is used:
    def call(self, inputs):        
        # Get the individual parameters of the model, here l1 and l2
        l1 = inputs[:, 0:1]
        l2 = inputs[:, 2:3]
        
        # Do the forward-feed
        output_1 = tf.matmul(l1, self.weights_1) + self.biases[0]
        output_2 = tf.matmul(inputs, self.weights_2) + self.biases[1]
        output_3 = tf.matmul(l2, self.weights_3) + self.biases[2]
        
        # Apply the activation function: here 'ReLU'
        output_2 = tf.nn.relu(output_2) + 10 ** -14
        
        # Apply the custom activation function ensure m1 > sqrt(3/8*(l1-l2)^2) and to compute the strut length
        output_2 = self.strut_activation(output_1, output_2, output_3)
        
        # Shape the output correctly
        output = tf.concat([output_1, output_2, output_3], axis=1)
        return output

class NN(tf.keras.Model):
    def __init__(self, N1=10, N3=5, lamb=1e-2, **kwargs):
        super(NN, self).__init__(self, **kwargs)
        
        # Make the hidden layer callable as 'self.hidden'
        self.hidden = HiddenLayer()
        
        # Define global constants for the model in 'float64' format
        self.pi = tf.constant(np.pi, dtype=tf.float64)
        self.one = tf.constant(1., dtype=tf.float64)
        self.minus_one = tf.constant(-1., dtype=tf.float64)
        
        # Define Regge calculus parameters
        self.N1 = tf.constant(N1, dtype=tf.float64)
        self.N3 = tf.constant(N3, dtype=tf.float64)
        self.lamb = tf.constant(lamb, dtype=tf.float64)

        if N3 == 600:
            self.nte = tf.constant(5, dtype=tf.float64)
        elif N3 == 16:
            self.nte = tf.constant(4, dtype=tf.float64)
        else:
            self.nte = tf.constant(3, dtype=tf.float64)
        
        # Define variables to find the minimal loss later
        self.loss_array = []
        self.argmin = None
        self.minimum_loss = None      

    def get_params(self):
        print("N1 = ", self.N1)
        print("N3 = ", self.N3)
        print("lambda = ", self.lamb)
    
    # Forward-feed when the model is used
    def call(self, inputs):
        # Convert inputs to dtype=tf.float64
        inputs = tf.cast(inputs, dtype=tf.float64)
        
        # Only the hidden layer is called and its forward-feed will be executed
        return self.hidden(inputs)
        
    # Set the argument range of 'tf.acos' to [-1., 1.] since numeric computations could
    # lead to values that are slightly outside this range due to computational error
    def crop(self, arg):
        # Define the conditions
        too_small = arg < self.minus_one
        too_large = arg > self.one

        # Use tensorflow's if/then/else: (if condition, then, else(if condition2, then, else))
        output = tf.where(too_small, self.minus_one, tf.where(too_large, self.one, arg))
        return output
    
    # Get trainable weights of a pre-trained model and hand it to the hidden layer
    def set_weights(self, weights):
        weights = tf.cast(weights, dtype=tf.float64)
        self.hidden.set_weights(weights)
    
    # Output the trainable weights of the hidden layer
    def get_weights(self):
        return self.hidden.trainable_weights
    
    def compute_loss(self, prediction):
        # Define the parts of the output in Regge calculus variables            
        l1 = prediction[0,0]
        m1 = prediction[0,1]
        l2 = prediction[0,2]

        # Compute the value of the EOMm: the first fraction
        numerator1 = -(l1 + l2) * (tf.math.square(l1) + tf.math.square(l2)) * self.lamb * m1 * self.N3
        denominator1 = tf.constant(12,dtype=tf.float64) * tf.math.sqrt(tf.constant(-3,dtype=tf.float64) * tf.math.square(l1 - l2) + tf.constant(8,dtype=tf.float64) * tf.math.square(m1))

        # Compute the value of the EOMm: the second fraction
        # 'self.crop' ensures the arguments of the arccos to be indeed in [-1, 1]
        arg = self.crop( (tf.math.square(l1 - l2) - tf.constant(2,dtype=tf.float64) * tf.math.square(m1)) / (tf.constant(2,dtype=tf.float64) * tf.math.square(l1 - l2) - tf.constant(6,dtype=tf.float64) * tf.math.square(m1)) )
        numerator2 = (l1 + l2) * m1 * self.N1 * (tf.constant(2,dtype=tf.float64) * self.pi - self.nte * tf.math.acos(arg) )
        denominator2 = tf.math.sqrt(- tf.math.square(l1 - l2) + tf.constant(4,dtype=tf.float64) * tf.math.square(m1))

        # Define the total loss as: EOMm ^ 2. This ensures that the solution of the time step
        # can be found as a minimizing procedure, since loss = 0 <--> EOMm = 0 i.e., a classical solution
        loss = tf.math.square(numerator1 / denominator1 + numerator2 / denominator2)
        return loss
    
    def training(self, inputs, epochs, rl=False):
        # Convert inputs to dtype=tf.float64
        inputs = tf.cast(inputs, dtype=tf.float64)
        
        # Initialize loss tracking parameters for repetitive trainings
        self.loss_array = []
        self.argmin = None
        self.minimum_loss = None
        
        for i in tqdm(range(epochs)):
            self.loss_array.append(self.train_step(inputs)["loss"].numpy())
            
        if len(self.loss_array)==0:
            self.loss_array.append(self.compute_loss(inputs)) # To make one epoch possible
        
        self.argmin = np.argmin(self.loss_array)
        self.minimum_loss = self.loss_array[self.argmin]
        
        if rl:
            return self.argmin, self.minimum_loss, self.loss_array
        else:
            return self.argmin, self.minimum_loss
    
    # Define the training procedure
    # '@tf.function' speeds up training incredibly: --- DO NOT REMOVE! ---
    @tf.function
    def train_step(self, inputs):        
        # Use the automatic gradient computation        
        with tf.GradientTape(persistent=True) as tape:
            # Perform 1 forward-feed step
            prediction = self(inputs, training=True)  # Forward pass
            
            loss = self.compute_loss(prediction)
                    
        # Tell tensorflows automatic gradient computation to compute the gradients
        # of the loss with respect to the trainable variables of the network:
        # here only the three weights of the strut-neuron in the hidden layer
        gradients = tape.gradient(loss, self.trainable_variables)

        # Apply the gradients to update the weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss}

# Define the hidden layer of the neural network
class HiddenLayer2(tf.keras.layers.Layer):
    # Initialize the hidden layer.
    def __init__(self):
        super(HiddenLayer2, self).__init__()

        self.pi = tf.constant(np.pi, dtype=tf.float64)
        
        # Define the weights for the three neurons respectively
        weights_1 = tf.constant([[1.]], dtype=tf.float64)
        weights_2 = tf.constant([[0.], [1.], [0.]], dtype=tf.float64)
        weights_3 = tf.constant([[1.]], dtype=tf.float64)
        
        # The biases are set to be '0.' as they do not influence the spatial edges in this case
        biases = tf.constant([0., 0., 0.], dtype=tf.float64)
        
        # Set the weights for the hidden layer:
        # the weights for the boundary spatial edges and the biases are set to be non-trainable
        # whereas the weights for the strut can be learned
        self.weights_1 = tf.Variable(initial_value=weights_1, trainable=False, dtype=tf.float64)
        self.weights_2 = tf.Variable(initial_value=weights_2, trainable=True, dtype=tf.float64)
        self.weights_3 = tf.Variable(initial_value=weights_3, trainable=False, dtype=tf.float64)
        self.biases = tf.Variable(initial_value=biases, trainable=False, dtype=tf.float64)
    
    # Set the trainable weights to the values of a pre-trained model
    def set_weights(self, weights_2):
        self.weights_2.assign(weights_2)
        
    # Define an activation function for the strut-neurons
    def strut_activation(self, x):
        return tf.math.tan(x)
    
    # Forward-feed when the hidden layer is used:
    def call(self, inputs):        
        # Get the individual parameters of the model, here l1 and l2
        l1 = inputs[:, 0:1]
        l2 = inputs[:, 2:3]
        
        # Do the forward-feed
        output_1 = tf.matmul(l1, self.weights_1) + self.biases[0]
        output_2 = tf.matmul(inputs, self.weights_2) + self.biases[1]
        output_3 = tf.matmul(l2, self.weights_3) + self.biases[2]
        
        # Apply the activation function: here tan(x)
        output_2 = tf.where(output_2 > self.pi, self.pi-10**-14, output_2)
        output_2 = self.strut_activation(output_2)
        
        # Shape the output correctly
        output = tf.concat([output_1, output_2, output_3], axis=1)
        return output
    
class NN2(tf.keras.Model):
    def __init__(self, N1=10, N3=5, lamb=1e-2, **kwargs):
        super(NN2, self).__init__(self, **kwargs)
        
        # Make the hidden layer callable as 'self.hidden'
        self.hidden = HiddenLayer2()
        
        # Define global constants for the model in 'float64' format
        self.pi = tf.constant(np.pi, dtype=tf.float64)
        self.one = tf.constant(1., dtype=tf.float64)
        self.minus_one = tf.constant(-1., dtype=tf.float64)
        
        # Define Regge calculus parameters
        self.N1 = tf.constant(N1, dtype=tf.float64)
        self.N3 = tf.constant(N3, dtype=tf.float64)
        self.lamb = tf.constant(lamb, dtype=tf.float64)

        if N3 == 600:
            self.nte = tf.constant(5, dtype=tf.float64)
        elif N3 == 16:
            self.nte = tf.constant(4, dtype=tf.float64)
        else:
            self.nte = tf.constant(3, dtype=tf.float64)
        
        # Define variables to find the minimal loss later
        self.loss_array = []
        self.argmin = None
        self.minimum_loss = None     

    def get_params(self):
        print("N1 = ", self.N1)
        print("N3 = ", self.N3)
        print("lambda = ", self.lamb)
    
    # Forward-feed when the model is used
    def call(self, inputs):
        # Convert inputs to dtype=tf.float64
        inputs = tf.cast(inputs, dtype=tf.float64)
        
        # Only the hidden layer is called and its forward-feed will be executed
        return self.hidden(inputs)
        
    # Set the argument range of 'tf.acos' to [-1., 1.] since numeric computations could
    # lead to values that are slightly outside this range due to computational error
    def crop(self, arg):
        # Define the conditions
        too_small = arg < self.minus_one
        too_large = arg > self.one

        # Use tensorflow's if/then/else: (if condition, then, else(if condition2, then, else))
        output = tf.where(too_small, self.minus_one, tf.where(too_large, self.one, arg))
        return output
    
    # Get trainable weights of a pre-trained model and hand it to the hidden layer
    def set_weights(self, weights):
        weights = tf.cast(weights, dtype=tf.float64)
        self.hidden.set_weights(weights)
    
    # Output the trainable weights of the hidden layer
    def get_weights(self):
        return self.hidden.trainable_weights
    
    def compute_loss(self, prediction):
        # Define the parts of the output in Regge calculus variables            
        l1 = prediction[0,0]
        m1 = prediction[0,1]
        l2 = prediction[0,2]

        # Compute the value of the EOMm: the first fraction
        numerator1 = -(l1 + l2) * (tf.math.square(l1) + tf.math.square(l2)) * self.lamb * m1 * self.N3
        denominator1 = tf.constant(12,dtype=tf.float64) * tf.math.sqrt(tf.constant(-3,dtype=tf.float64) * tf.math.square(l1 - l2) + tf.constant(8,dtype=tf.float64) * tf.math.square(m1))

        # Compute the value of the EOMm: the second fraction
        # 'self.crop' ensures the arguments of the arccos to be indeed in [-1, 1]
        arg = self.crop( (tf.math.square(l1 - l2) - tf.constant(2,dtype=tf.float64) * tf.math.square(m1)) / (tf.constant(2,dtype=tf.float64) * tf.math.square(l1 - l2) - tf.constant(6,dtype=tf.float64) * tf.math.square(m1)) )
        numerator2 = (l1 + l2) * m1 * self.N1 * (tf.constant(2,dtype=tf.float64) * self.pi - self.nte * tf.math.acos(arg) )
        denominator2 = tf.math.sqrt(- tf.math.square(l1 - l2) + tf.constant(4,dtype=tf.float64) * tf.math.square(m1))

        # Define the total loss as: EOMm ^ 2. This ensures that the solution of the time step
        # can be found as a minimizing procedure, since loss = 0 <--> EOMm = 0 i.e., a classical solution
        loss = tf.math.square(numerator1 / denominator1 + numerator2 / denominator2)
        return loss
    
    def training(self, inputs, epochs, rl=False):
        # Convert inputs to dtype=tf.float64
        inputs = tf.cast(inputs, dtype=tf.float64)
        
        # Initialize loss tracking parameters for repetitive trainings
        self.loss_array = [] 
        self.argmin = None
        self.minimum_loss = None
        
        for i in tqdm(range(epochs)):
            self.loss_array.append(self.train_step(inputs)["loss"].numpy())
            
        if len(self.loss_array)==0:
            self.loss_array.append(self.compute_loss(inputs)) # To make one epoch possible
        
        self.argmin = np.argmin(self.loss_array)
        self.minimum_loss = self.loss_array[self.argmin]
        
        if rl:
            return self.argmin, self.minimum_loss, self.loss_array
        else:
            return self.argmin, self.minimum_loss
    
    # Define the training procedure
    # '@tf.function' speeds up training incredibly: --- DO NOT REMOVE! ---
    @tf.function
    def train_step(self, inputs):        
        # Use the automatic gradient computation        
        with tf.GradientTape(persistent=True) as tape:
            # Perform 1 forward-feed step
            prediction = self(inputs, training=True)  # Forward pass
            
            loss = self.compute_loss(prediction)
                    
        # Tell tensorflows automatic gradient computation to compute the gradients
        # of the loss with respect to the trainable variables of the network:
        # here only the three weights of the strut-neuron in the hidden layer
        gradients = tape.gradient(loss, self.trainable_variables)

        # Apply the gradients to update the weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss}

def OneStepSolver(inputs, N1=10, N3=5, lamb=0.1, epochs=5000, lr=1e-6, rtw=False, graph=False, adam_beta_1=0.9, adam_beta_2=0.999, adam_epsilon=1e-07, adam_amsgrad=False, adam_decay=0.9, adam_clipnorm=1., adam_clipvalue=None, adam_global_clipnorm=None, repetition=4, lr_mult=5, lr_add=5e-1, lr_limit=5e-1, loss_accuracy=1e-26, epochs_mult=2, weights=[[0.],[1.],[0.]], **kwargs):    
    params = {
        'N1': N1,
        'N3': N3,
        'lamb': lamb
    }
    
    user_input = "y"
    
    if not consistency_check(N1, N3, lamb, l1=inputs[0,0].numpy(), l2=inputs[0,2].numpy(), decimals=0):
        print("There is no solution! Press 'Enter' to run the neural network anyway or 'no' to abort ...")
        user_input = input()  # Wait for keyboard interaction

    if user_input.lower() == 'n' or user_input.lower() == 'no':
        print("Code aborted.")
    else:
        count = 0
        lossmin = 1
        multiplied = False
        added = False

        while lossmin > loss_accuracy and count < repetition:
            tf.print("--- lambda = {} ---".format(lamb))

            #--------------------Create an instance of the model, define the optimizer and compile it-------
            model = NN(**params)
            adam_optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr,
                beta_1=adam_beta_1,
                beta_2=adam_beta_2,
                epsilon=adam_epsilon,
                amsgrad=adam_amsgrad,
                decay=adam_decay,
                clipnorm=adam_clipnorm,
                clipvalue=adam_clipvalue,
                global_clipnorm=adam_global_clipnorm)
            model.compile(optimizer=adam_optimizer)

            # Get the trainable weights of the hidden layer from a pre-trained model
            model.set_weights(weights)

            # Train the model
            pos, lossmin = model.training(inputs, epochs=epochs)
            pos = np.copy(pos)

            count +=1
            if lr <= lr_limit:
                lr *= lr_mult
                multiplied = True
                added = False
            else:
                lr += lr_add
                epochs *= epochs_mult
                multiplied = False
                added = True

        if multiplied and not added:
            lr = lr / lr_mult
        elif not multiplied and added:
            lr -= lr_add

        if lossmin > loss_accuracy:
            tf.print("Failed to reach accuracy with maximum learning rate ", lr)
            tf.print("Mabye a larger learning rate works ... ")

        tf.print("learning rate: ", lr)
        tf.print("minimum loss: ", lossmin)

        # Do a last iteration only up to the position of the minimum
        model = NN(**params)
        adam_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=adam_beta_1,
            beta_2=adam_beta_2,
            epsilon=adam_epsilon,
            amsgrad=adam_amsgrad,
            decay=adam_decay,
            clipnorm=adam_clipnorm,
            clipvalue=adam_clipvalue,
            global_clipnorm=adam_global_clipnorm)
        model.compile(optimizer=adam_optimizer)
        model.set_weights(weights)

        pos, lossmin, losses = model.training(inputs, epochs=pos, rl=True)
        if len(losses)>1:
            losses.append(model.compute_loss(model(inputs))) # To see the minimal loss actually in the array

        losses = np.copy(losses)
        trained_weights = model.get_weights()[0]

        # Convert tf-struts to numpy value
        out = model(inputs)
        outputs = []
        for i in range(len(out[0])):
            outputs.append(out[0,i].numpy())

        tf.print("output = ", outputs)

        if graph:
            plt.loglog(np.arange(len(losses))+1, losses)
            plt.grid()
            plt.grid(which='minor')
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Training behavior")
            plt.show()

        if rtw:
            return outputs, trained_weights, losses
        else:
            return outputs, losses

def OneStepSolver2(inputs, N1=10, N3=5, lamb=17.583, epochs=5000, lr=1e-1, rtw=False, graph=False, adam_beta_1=0.9, adam_beta_2=0.999, adam_epsilon=1e-07, adam_amsgrad=False, adam_decay=0.9, adam_clipnorm=1e-2, adam_clipvalue=None, adam_global_clipnorm=None, weights=[[0.],[1.],[0.]], **kwargs):
    # Set keyword arguments for training
    training_kwargs = {
        'N1': N1,
        'N3': N3,
        'lamb': lamb,
    }
    
    user_input = "y"
    
    if not consistency_check(N1, N3, lamb, l1=inputs[0,0].numpy(), l2=inputs[0,2].numpy(), decimals=0):
        print("There is no solution! Press 'Enter' to run the neural network anyway or 'no' to abort ...")
        user_input = input()  # Wait for keyboard interaction

    if user_input.lower() == 'n' or user_input.lower() == 'no':
        print("Code aborted.")
    else:  
        tf.print("--- lambda = {} ---".format(lamb))

        model2 = NN2(**training_kwargs)
        adam_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=adam_beta_1,
            beta_2=adam_beta_2,
            epsilon=adam_epsilon,
            amsgrad=adam_amsgrad,
            decay=adam_decay,
            clipnorm=adam_clipnorm,
            clipvalue=adam_clipvalue,
            global_clipnorm=adam_global_clipnorm)
        model2.compile(optimizer=adam_optimizer)

        # Train the model
        pos, lossmin = model2.training(inputs, epochs=epochs)

        tf.print("learning rate: ", lr)
        tf.print("minimum loss: ", lossmin)

        # Second iteration to train only up to the minimum
        model2 = NN2(**training_kwargs)
        adam_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=adam_beta_1,
            beta_2=adam_beta_2,
            epsilon=adam_epsilon,
            amsgrad=adam_amsgrad,
            decay=adam_decay,
            clipnorm=adam_clipnorm,
            clipvalue=adam_clipvalue,
            global_clipnorm=adam_global_clipnorm)
        model2.compile(optimizer=adam_optimizer)

        # Train the model
        pos, lossmin, losses = model2.training(inputs, epochs=pos, rl=True)

        if len(losses)>1:
            losses.append(model2.compute_loss(model2(inputs))) # To see the minimal loss actually in the array

        losses = np.copy(losses)
        trained_weights = model2.get_weights()[0]

        # Convert tf-struts to numpy value
        out = model2(inputs)
        outputs = []
        for i in range(len(out[0])):
            outputs.append(out[0,i].numpy())
            if outputs[i] < 0:
                outputs[i] *= -1

        tf.print("output = ", outputs)

        if graph:
            plt.loglog(np.arange(len(losses))+1, losses)
            plt.grid()
            plt.grid(which='minor')
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Training behavior")
            plt.show()

        if rtw:
            return outputs, trained_weights, losses
        else:
            return outputs, losses

#Compute the linspaced array of possible values of lambda due to the EOMm
def lambda_array(N1, N3, l1, l2, lin=1, log=5):
    # lin: accuracy in the linear spacings
    # log: accuracy in the logarithmic spacings
    
    limit = lambda_limit(N1, N3, l1, l2)
    pos = int(oom(limit) - (lin+1))
    stepsize = 10 ** pos
    steps = int(np.trunc(limit/stepsize))
    tmp1 = np.logspace(pos-log, pos, log, endpoint=False)
    tmp2 = np.linspace(stepsize, np.trunc(limit/stepsize)*stepsize, steps)
    return np.concatenate([tmp1, tmp2], axis=0)

def choose_solver(inputs, N1=10, N3=5, lamb=1e-3, **kwargs):
    # Define the parts of the output in Regge calculus variables            
    l1 = inputs[0,0]
    m1 = inputs[0,1]
    l2 = inputs[0,2]
    
    if consistency_check(N1, N3, lamb, l1, l2):
        limit = lambda_limit(N1, N3, l1, l2)

        if lamb < 0.96 * limit:
            print("OneStepSolver")
        else:
            print("OneStepSolver2")