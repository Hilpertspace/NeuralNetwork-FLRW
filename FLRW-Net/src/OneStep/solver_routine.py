"""This module implements the solver routine."""
from pathlib import Path

import sys

import tkinter as tk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText

import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from OneStep.neural_network import Network

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from user_interface import save_weights_dialog
from user_interface import save_output_dialog
from user_interface import save_graph_dialog
from user_interface import config_training_window

def on_closing_training_window(window, fig, finished):
    """Close the figure upon closing the training window."""
    if not finished:
        messagebox.showwarning("Warning",
        "Please abort the training via the according button.")
        window.lift()
    else:
        plt.close(fig)
        window.destroy()

def on_closing_root_window_training(root_window, training_window, fig, finished_training):
    """Set closing dialog."""
    if not finished_training:
        try:
            on_closing_training_window(training_window, fig, finished_training)
        except ValueError:
            tf.print("")
    else:
        if messagebox.askokcancel("Quit", "Do you really want to quit FLRW-Net's user interface?"):
            try:
                plt.close(fig)
            except ValueError:
                tf.print("")

            messagebox.showinfo("Goodbye", "Hope to see you soon :)")
            root_window.destroy()

def one_step_solver(root_window, inputs, n1=10, n3=5, nte=3, lamb=0.1, epochs=5000, lr=1e-6,
                   adam_clipnorm=1., loss_accuracy=1e-26, weights=[[0.],[1.],[0.]]):
    """
    Final function that handles all the individual parts from closing events and calling the network to generating the output data.
    """

    # Set up the training window
    training_window = tk.Toplevel(root_window)
    training_window.withdraw()

    # Create a Matplotlib figure and plot the data
    fig, ax = plt.subplots()

    # Set finished training flag
    finished_training = False

    # Close the figure upon closing the training window
    training_window.protocol("WM_DELETE_WINDOW",
        lambda: on_closing_training_window(training_window, fig, finished_training))

    # Upon closing the root window ask whether FLRW-Net's UI should really be closed.
    root_window.protocol("WM_DELETE_WINDOW",
        lambda: on_closing_root_window_training(root_window,training_window,fig,finished_training))

    # Create the ScrolledText widget to use as output
    output_text = ScrolledText(training_window, width=70, height=13)

    # Configure the training window
    config_training_window(training_window, output_text)
    training_window.deiconify()

    params = {
        'n1': n1,
        'n3': n3,
        'nte': nte,
        'lamb': lamb,
        'loss_threshold': loss_accuracy
    }

    start_time = time.time()

    #-------Create an instance of the model, define the optimizer and compile it-------
    model = Network(training_window, **params)
    adam_optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        decay=0.9,
        clipnorm=adam_clipnorm,
        clipvalue=None,
        global_clipnorm=None)
    model.compile(optimizer=adam_optimizer)

    # Set the trainable weights of the hidden layer to the ones of a pre-trained model
    model.set_weights(weights)

    # Train the model
    lossmin, losses = model.training(inputs, epochs=epochs)

    end_time = time.time()

    print(f"Execution time: {end_time - start_time: .1f} seconds.")

    # Terminate one_step_solver if the training has been aborted
    if lossmin is None:
        finished_training = True
        return None

    # Get the weigths that produced the minimum loss
    trained_weights = model.get_min_weights()[0]

    # Convert tf-weights to numpy value
    np_trained_weights = []
    for i in range(tf.size(trained_weights).numpy()):
        np_trained_weights.append(trained_weights[i,0].numpy())

    # Set min weights and compute the according value of the struts
    model.set_weights(trained_weights)
    out = model(inputs)

    # Convert tf-struts to numpy value
    outputs = []
    for i in range(len(out[0])):
        outputs.append(out[0,i].numpy())

    # Print the output and disable the ScrolledText widget
    print("output = ", outputs)
    print("minimum loss = ", lossmin)
    print(f"trained_weights = {np_trained_weights}")
    output_text.config(state=tk.DISABLED)

    finished_training = True

    # Configure the training graph
    ax.loglog(np.arange(len(losses))+1, losses)
    ax.grid(True, which='both')  # Show both major and minor grid lines
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Training Behavior")

    # Create a Canvas for the Matplotlib graph
    graph_frame = tk.Frame(training_window, width=500, height=350)
    graph_frame.grid(row=2, column=0, columnspan=4, padx=0, pady=5)
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Overwrite buttons for the training window
    save_weights_btn = tk.Button(training_window, text="Save weights", background="lightgray",
                        command=lambda: save_weights_dialog(training_window, np_trained_weights))
    save_weights_btn.grid(row=1, column=1, columnspan=1, padx=5, pady=5)

    save_output_btn = tk.Button(training_window, text="Save output", background="lightgray",
                                   command=lambda: save_output_dialog(training_window, outputs))
    save_output_btn.grid(row=1, column=2, columnspan=1, padx=5, pady=5)

    save_graph_btn = tk.Button(training_window, text="Save graph", background="lightgray",
                                  command=lambda: save_graph_dialog(training_window, fig))
    save_graph_btn.grid(row=1, column=3, columnspan=1, padx=5, pady=5)

    save_weights_btn.config(state="normal")
    save_output_btn.config(state="normal")
    save_graph_btn.config(state="normal")

def run_FLRW_Net(root,n1,n3,nte,lamb,epochs,lr,clipnorm,loaded_weights,ini,fin,loss_accuracy):
    """Configure and run FLRW-Net."""
    params = {
    'n1': n1,
    'n3': n3,
    'nte': nte,
    'lamb': lamb,
    'epochs': epochs,
    'lr': lr,
    'adam_clipnorm': clipnorm,
    'loss_accuracy': loss_accuracy
    }

    # Set initial guess for the parameter related to the strut
    if n3==5:
        a1 = 2./5 - 3./8
    elif n3==16:
        a1 = 1./2 - 3./8
    else:
        a1 = (3+np.sqrt(5)) / 2

    x=tf.constant([[ini, a1, fin]],dtype=tf.float64)

    # Set standard precision from float32 to float64
    tf.keras.backend.set_floatx('float64')

    if loaded_weights is not None:
        one_step_solver(root, inputs=x, weights=loaded_weights, **params)
    else:
        one_step_solver(root, inputs=x, **params)
