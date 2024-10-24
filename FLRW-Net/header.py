"""This module stores general functions that fit in neither category."""

import sys
import tkinter as tk
from tkinter import messagebox

import numpy as np

def redirect_output_to_widget(widget):
    """
    Switch the standard output to the scrolled text widget in the training window.
    Standard output and error prompts are redirected.
    """
    class StdoutRedirector:
        """
        Define the output redirector.
        """
        def __init__(self, widget):
            self.widget = widget
            self.linebuf = ''

        def write(self, message):
            """
            If the output is from a tqdm progress bar ('\r' at the end), the line should be
            overwritten in the next iteration. This creates a progress bar in a single line
            inside the scrolled text widget.
            """
            #sys.__stdout__.write(f"DEBUG: Writing message: {repr(message)}")

            if '\r' in message:
                self.widget.delete("end-1c linestart", "end-1c lineend")
            self.widget.insert(tk.END, message)
            self.widget.see(tk.END)

        def flush(self):
            """
            Necessary to specify in this class (avoid too few public methods error). Not used.
            """
            #sys.__stdout__.write("DEBUG: flush() called")
            pass

    # Redirect the standard ouput
    sys.stdout = StdoutRedirector(widget)

    # Redirect the standard error (tqdm progress bar prints here)
    sys.stderr = StdoutRedirector(widget)

def compute_spatial_edges(sf, n3):
    """Compute the spatial edge lengths if scale factors (sf) are prescribed as boundary data."""
    edge_length = sf * np.power(12 * np.sqrt(2) * np.power(np.pi,2.) / n3, 1./3)
    return edge_length

def compute_lambda_limit(nte):
    """
    Compute an upper boundary related to the maximum value of the cosmological constant for which
    there is a solution given the prescribed boundary data.
    """
    return 12 * np.sqrt(2) * (2*np.pi - nte*np.arccos(1/3))

def compute_value(n1, n3, lamb, ini, fin):
    """
    Compute a value related to the cosmological constant to determine whether it lies above the
    maximum value for which there is a solution.
    """
    return lamb * (ini**2 + fin**2) * n3 / n1

def set_triangulation_params(triangulation):
    """Set the parameters that specify the spatial triangulations."""
    warning = False

    if triangulation=="5-cell":
        n1 = 10
        n2 = 10
        n3 = 5
        nte = 3
    elif triangulation=="16-cell":
        n1 = 24
        n2 = 32
        n3 = 16
        nte = 4
    elif triangulation=="600-cell":
        n1 = 720
        n2 = 1200
        n3 = 600
        nte = 5
    else:
        n1 = None
        n2 = None
        n3 = 1
        nte = None
        warning = True

    return n1, n2, n3, nte, warning

def set_boundary_params(boundary, entry2, entry3, n3):
    """Set the parameters that specify the boundary data."""
    warning = False

    if boundary.get()=="edge lengths":
        ini = float(entry2.get())
        fin = float(entry3.get())
    elif boundary.get()=="scale factors":
        tmp_ini = float(entry2.get())
        tmp_fin = float(entry3.get())
        ini = compute_spatial_edges(tmp_ini, n3)
        fin = compute_spatial_edges(tmp_fin, n3)
    else:
        ini = None
        fin = None
        warning = True

    return ini, fin, warning

def handle_warning_messages(triangulation_warning, boundary_warning, triangulation, compute_button,
                            entry1, entry4, entry5, entry6, n1, n3, nte, ini, fin):
    """Implementation of the warning messages for the different cases."""
    lamb = None
    epochs = None
    lr = None
    clipnorm = None

    if triangulation_warning and boundary_warning:
        messagebox.showwarning("Warning", "Please choose a triangulation and the type of "
                                "boundary data that you prescribe!")
    elif triangulation_warning:
        messagebox.showwarning("Warning", "Please choose a triangulation!")
    elif boundary_warning:
        messagebox.showwarning("Warning", "Please choose the type of boundary data that "
                                "you prescribe!")
    else:
        lamb = float(entry1.get())
        epochs = int(entry4.get())
        lr = float(entry5.get())
        clipnorm = float(entry6.get())

        if compute_value(n1, n3, lamb, ini, fin) >= compute_lambda_limit(nte):
            messagebox.showwarning("Warning", "There is no solution to these boundary data"
                                    " for the prescribed value of the cosmological "
                                    "constant!\n\nSaved inputs anyway.")
        else:
            message = (
                f"Saved input for the {triangulation.get()} model with:\n\n"
                f"lambda={lamb}\n"
                f"ini={ini}\n"
                f"fin={fin}\n"
                f"epochs={epochs}\n"
                f"learning rate={lr}\n"
                f"clipnorm={clipnorm}"
            )

            messagebox.showinfo("Confirmation", message)

        compute_button.config(state="normal")

    return lamb, epochs, lr, clipnorm
