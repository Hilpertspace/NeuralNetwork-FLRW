"""This module implements FLRW-Net's UI."""

import json
import tkinter as tk
import webbrowser
from pathlib import Path

from tkinter import font, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np

import header as hdr

# Assign global variables with default value
LOADED_WEIGHTS = None
TRIANGULATION_VALUE = None
BOUNDARY_VALUE = None
WEIGHTS_VALUE = None
FLRW_CONFIG = None
STEPS_VALUE = None

N1 = None
N2 = None
N3 = None
NTE = None
LAMB = None
EPOCHS = None
LR = None
CLIPNORM = None
INI = None
FIN = None

# Set default values for the entry fields
DEFAULT_CC = "0.001"        # Entry 1: cosmological constant
DEFAULT_EDGE_INI = "1"      # Entry 2: initial boundary
DEFAULT_EDGE_FIN = "2"      # Entry 3: final boundary
DEFAULT_EPOCHS = "5000"     # Entry 4: number of EPOCHS
DEFAULT_LR = "0.001"        # Entry 5: learning rate
DEFAULT_CLIPNORM = "1"      # Entry 6: CLIPNORM of adam optimizer

UPDATED_CONFIG = False
CHANGED_STEPS = False
LOSS_THRESHOLD = 1e-26

def centering_window(window, width, height):
    """Center a specified window on the screen."""
    # Get the screen width and height
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    # Calculate the x and y coordinates to center the window
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    # Set the window's position
    window.geometry(f"{width}x{height}+{x}+{y}")

def purpose():
    """Show the purpose of FLRW-Net upon clicking on the according button."""
    messagebox.showinfo("Purpose", "A machine learning project in Python, using TensorFlow & "
                        "Keras, to compute the time-evolution of an FLRW universe in Euclidean "
                        "Regge calculus with discrete time steps.\n\nPlease specify the task by "
                        "chosing the number of time steps, the triangulation, the type of your boundary data and, if necessary, whether you want to use custom weights, e.g., weights from an earlier calculation, using the top"
                        " menu.")

def my_copyright():
    """Show the copyright statement."""
    messagebox.showinfo("Copyright", "Copyright 2024 Florian Emanuel Hilpert")

def open_link(window):
    """Open the link in a web browser."""
    webbrowser.open("https://github.com/Hilpertspace/NeuralNetwork-FLRW")
    window.destroy()

def git_message_box(root):
    """Show a message box containing the link the the Git project."""

    # Create a Toplevel window for the message box
    message_box = tk.Toplevel(root)
    message_box.title("Git project")
    message_box.lift()
    message_box.configure(bg="#FFFFFF") # "white == #FFFFFF"
    message_box.resizable(False, False)

    # Set the icon for the application
    message_box.iconbitmap("Icon.ico")

    # Set the size of the window
    window_width = 350
    window_height = 70

    centering_window(message_box, window_width, window_height)

    # Message text
    message = ("The code for this poject is available at:"
              "\nhttps://github.com/Hilpertspace/NeuralNetwork-FLRW")

    # Create a font object and configure it for underline on hover
    default_font = font.nametofont("TkDefaultFont")
    underlined_font = default_font.copy()
    underlined_font.configure(underline=True)

    # Create a label as a hyperlink
    label = ttk.Label(message_box, text=message, foreground="black",
                      background="#FFFFFF", cursor="hand2")
    label.pack(padx=10, pady=10)
    label.bind("<Button-1>", lambda event: open_link(message_box))
    label.bind("<Enter>", lambda event: label.config(font=underlined_font))
    label.bind("<Leave>", lambda event: label.config(font="TkDefaultFont"))

    # Display the message box
    message_box.grab_set()  # Make the message box modal
    message_box.mainloop()

def open_documentation():
    """Open the documentation in the user's default webbrowser."""
    # Define the path to your HTML file
    file_path = Path(".") / "docs/html/documentation.html"
    file_path = file_path.resolve()

    browser = webbrowser.get()

    # Open the HTML file in the default web browser
    browser.open(f'file://{file_path}')

def save_weights(training_window, dialog, fname, weights):
    """Save the trained weights of the network to a .json file."""
    # Define file path
    folder_path = Path(".") / "Weights"
    folder_path.mkdir(parents=True, exist_ok=True)
    filename = fname.get() + ".json"
    path = folder_path / filename

    if path.exists():
        messagebox.showwarning("File Exists", "The file already exists. "
                               "Please enter a different filename.")
        dialog.lift()
    else:
        # Create a dictionary with a user's data
        data = {
            "weights": [
                {
                    "weight 1": weights[0],
                    "weight 2": weights[1],
                    "weight 3": weights[2]
                }
            ]
        }

        # Write the dictionary to the JSON file
        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

        messagebox.showinfo("Success", "Saved weights.")
        dialog.destroy()
        training_window.lift()

def save_weights_dialog(window, np_trained_weights):
    """Show saving weights window upon clicking the according button in the training window."""

    # Create a new Toplevel window
    dialog = tk.Toplevel(window)
    dialog.withdraw()

    # Configure the dialog window
    dialog.title("Save weights")
    dialog.config(bg="white")
    dialog.lift()
    dialog.iconbitmap("Icon.ico")
    dialog.resizable(False, False)

    centering_window(dialog, 300, 150)
    dialog.deiconify()

    # Create and place the entry widget
    entry_label = tk.Label(dialog, text="Enter filename:")
    entry_label.config(bg="white")
    entry_label.pack(pady=10)

    # Create a frame to hold the entry and the label
    entry_frame = tk.Frame(dialog)
    entry_frame.pack(pady=10)

    # Create and place the entry widget inside the frame
    fname = tk.Entry(entry_frame)
    fname.pack(side=tk.LEFT)

    # Create and place the .json label inside the frame
    json_label = tk.Label(entry_frame, text=".json")
    json_label.config(bg="white")
    json_label.pack(side=tk.LEFT)

    # Create and place the submit button
    submit_button = tk.Button(dialog, text="Save", background="lightgray",
                              command=lambda: save_weights(window, dialog, fname, np_trained_weights))
    submit_button.pack(pady=10)

def save_output(training_window, dialog, fname, output):
    """Save the network's output to a .json file."""
    # Define file path
    folder_path = Path(".") / "Outputs"
    folder_path.mkdir(parents=True, exist_ok=True)
    filename = fname.get() + ".json"
    path = folder_path / filename

    if path.exists():
        messagebox.showwarning("File Exists", "The file already exists. "
                               "Please enter a different filename.")
        dialog.lift()
    else:
        # Create a dictionary with a user's data
        data = {
            "outputs": [
                {
                    "edge 1": output[0],
                    "strut 1": output[1],
                    "edge 2": output[2]
                }
            ]
        }

        # Write the dictionary to the JSON file
        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

        messagebox.showinfo("Success", "Saved output.")
        dialog.destroy()
        training_window.lift()

def save_output_dialog(window, output):
    """Show saving output window upon clicking the according button in the training window."""

    # Create a new Toplevel window
    dialog = tk.Toplevel(window)
    dialog.withdraw()

    # Configure the dialog window
    dialog.title("Save output")
    dialog.config(bg="white")
    dialog.lift()
    dialog.iconbitmap("Icon.ico")
    dialog.resizable(False, False)

    centering_window(dialog, 300, 150)
    dialog.deiconify()

    # Create and place the entry widget
    entry_label = tk.Label(dialog, text="Enter filename:")
    entry_label.config(bg="white")
    entry_label.pack(pady=10)

    # Create a frame to hold the entry and the label
    entry_frame = tk.Frame(dialog)
    entry_frame.pack(pady=10)

    # Create and place the entry widget inside the frame
    fname = tk.Entry(entry_frame)
    fname.pack(side=tk.LEFT)

    # Create and place the .json label inside the frame
    json_label = tk.Label(entry_frame, text=".json")
    json_label.config(bg="white")
    json_label.pack(side=tk.LEFT)

    # Create and place the submit button
    submit_button = tk.Button(dialog, text="Save", background="lightgray",
                            command=lambda: save_output(window, dialog, fname, output))
    submit_button.pack(pady=10)

def save_graph(training_window, dialog, fname, fig):
    """Save the graph containing the network's training behaviour to a .png file."""
    # Define file path
    folder_path = Path(".") / "Graphs"
    folder_path.mkdir(parents=True, exist_ok=True)
    filename = fname.get() + ".png"
    path = folder_path / filename

    if path.exists():
        messagebox.showwarning("File Exists", "The file already exists. "
                               "Please enter a different filename.")
        dialog.lift()
    else:
        fig.savefig(path)
        plt.close(fig)

        messagebox.showinfo("Success", "Saved graph.")
        dialog.destroy()
        training_window.lift()

def save_graph_dialog(window, fig):
    """Show saving graph window upon clicking the according button in the training window."""

    # Create a new Toplevel window
    dialog = tk.Toplevel(window)
    dialog.withdraw()

    # Configure the dialog window
    dialog.title("Save graph")
    dialog.config(bg="white")
    dialog.lift()
    dialog.iconbitmap("Icon.ico")
    dialog.resizable(False, False)

    centering_window(dialog, 300, 150)
    dialog.deiconify()

    # Create and place the entry widget
    entry_label = tk.Label(dialog, text="Enter filename:")
    entry_label.config(bg="white")
    entry_label.pack(pady=10)

    # Create a frame to hold the entry and the label
    entry_frame = tk.Frame(dialog)
    entry_frame.pack(pady=10)

    # Create and place the entry widget inside the frame
    fname = tk.Entry(entry_frame)
    fname.pack(side=tk.LEFT)

    # Create and place the .png label inside the frame
    png_label = tk.Label(entry_frame, text=".png")
    png_label.config(bg="white")
    png_label.pack(side=tk.LEFT)

    # Create and place the submit button
    submit_button = tk.Button(dialog, text="Save", background="lightgray",
                              command=lambda: save_graph(window, dialog, fname, fig))
    submit_button.pack(pady=10)

def load_weights(window, fname, new_value):
    """Load custom initial trainable weights for the network from a specified file."""
    # Create a Path object for the file
    folder_path = Path(".") / "Weights"
    filename = fname.get() + ".json"
    path = folder_path / filename

    # Check if the file exists
    if not path.exists():
        messagebox.showwarning("File not found", "The file could not be found in the 'Weights' "
                               "directory. Please enter a different filename.")
        window.lift()
        return None

    else:
        # Load data from file
        with open(path, 'r') as file:
            data = json.load(file)

        # Get user dictionary from data
        weights = data.get('weights', [])

        global LOADED_WEIGHTS
        LOADED_WEIGHTS = np.array([
            [weights[0]['weight 1']],
            [weights[0]['weight 2']],
            [weights[0]['weight 3']]
        ])

        window.destroy()

        messagebox.showinfo("Success", "Loaded custom weights as initial weights."
                               "\nFrom now on they are used in every computation, "
                               "until others are loaded or the default weights are set.")
        update_flrw_config("input3", new_value)

def load_weights_dialog(window, value):
    """Set up load weights dialog window."""
    # Create a new Toplevel window
    dialog = tk.Toplevel(window)
    dialog.title("Load weights")
    dialog.config(bg="white")
    dialog.lift()
    dialog.iconbitmap("Icon.ico")
    dialog.resizable(False, False)

    centering_window(dialog, 300, 150)

    # Create and place the entry widget
    entry_label = tk.Label(dialog, text="Enter filename:")
    entry_label.config(bg="white")
    entry_label.pack(pady=10)

    # Create a frame to hold the entry and the label
    entry_frame = tk.Frame(dialog)
    entry_frame.pack(pady=10)

    # Create and place the entry widget inside the frame
    fname = tk.Entry(entry_frame)
    fname.pack(side=tk.LEFT)

    # Create and place the .png label inside the frame
    png_label = tk.Label(entry_frame, text=".json")
    png_label.config(bg="white")
    png_label.pack(side=tk.LEFT)

    # Create and place the load button
    load_button = tk.Button(dialog, text="Load", background="lightgray",
        command=lambda: load_weights(dialog, fname, value))
    load_button.pack(pady=10)

def set_default_weights():
    """Set the network's initial trainable weights to their default value."""
    global LOADED_WEIGHTS
    LOADED_WEIGHTS = None
    messagebox.showinfo("Success", "The initial weights have been set to their default value."
                        "\nFrom now on they are used in every computation, "
                        "until others are loaded.")

def config_training_window(window, output_widget):
    """Set up the training window."""
    # Configure the window's properties
    window.title("Training")
    window.lift()
    window.config(bg="white")
    window.resizable(False, False)
    window.iconbitmap("Icon.ico")

    # Center the training window on the screen
    centering_window(window, width=630, height=750)

    # Configure the ScrolledText widget
    output_widget.grid(row=0, column=0, columnspan=4, padx=5, pady=5)
    output_widget.config(state=tk.NORMAL)

    # Redirect the output to the ScrolledText widget
    hdr.redirect_output_to_widget(output_widget)

    # Set disabled buttons during training
    save_weights_btn = tk.Button(window, text="Save weights", background="lightgray")
    save_weights_btn.grid(row=1, column=1, columnspan=1, padx=5, pady=5)
    save_weights_btn.config(state="disabled")

    save_output_btn = tk.Button(window, text="Save output", background="lightgray")
    save_output_btn.grid(row=1, column=2, columnspan=1, padx=5, pady=5)
    save_output_btn.config(state="disabled")

    save_graph_btn = tk.Button(window, text="Save graph", background="lightgray")
    save_graph_btn.grid(row=1, column=3, columnspan=1, padx=5, pady=5)
    save_graph_btn.config(state="disabled")

def loading(loading_window, root_window, progress_bar, i=0):
    """Simulate a loading at startup with a ttk progress bar."""
    if i <= 100:
        # Update the progress bar and the loading window
        progress_bar['value'] = i
        loading_window.update_idletasks()

        # Schedule next iteration of load
        loading_window.after(15, loading, loading_window, root_window, progress_bar, i+1)

    else:
        # Close the loading window after loading completes
        loading_window.destroy()

        # Make root window visible again
        root_window.deiconify()

def startup_loading_window(root_window):
    """Set up the startup loading window."""

    load_window = tk.Toplevel()
    load_window.title("Loading...")
    load_window.lift()
    load_window.attributes('-topmost', True)
    load_window.config(bg="white")
    load_window.resizable(False, False)

    # Set the icon for the application
    load_window.iconbitmap("Icon.ico")

    # Call the centering_window function to center the window
    centering_window(load_window, width=400, height=80)

    # Create a progress bar
    progress = ttk.Progressbar(load_window, orient="horizontal", length=300, mode="determinate")
    progress.pack(pady=20)

    # Simulate the loading process
    loading(load_window, root_window, progress)

def update_flrw_config(input_type, value):
    """Update the label on the root window showing the current config of FLRW-Net."""
    global STEPS_VALUE

    if input_type == "input1":
        TRIANGULATION_VALUE.set(value)
    elif input_type == "input2":
        BOUNDARY_VALUE.set(value)
    elif input_type == "input3":
        WEIGHTS_VALUE.set(value)

    FLRW_CONFIG.config(text=f"You want to compute {STEPS_VALUE.get()} for the "
        f"{TRIANGULATION_VALUE.get()} model\n and to prescribe {BOUNDARY_VALUE.get()} "
        f"as boundary data using {WEIGHTS_VALUE.get()} weights.")

    global UPDATED_CONFIG
    UPDATED_CONFIG = True

def default_weights_chosen(new_value):
    """Default the trainable weights initially used by the network."""
    set_default_weights()
    update_flrw_config("input3", new_value)

# Function to update the button's event function
def update_button_command(root, button, time_steps):
    global DEFAULT_CC, DEFAULT_LR, DEFAULT_CLIPNORM, UPDATED_CONFIG, CHANGED_STEPS, STEPS_VALUE

    #Avoid circular imports
    from OneStep.solver_routine import run_FLRW_Net as run_FLRW_Net_1
    from TwoStep.solver_routine import run_FLRW_Net as run_FLRW_Net_2
    from ThreeStep.solver_routine import run_FLRW_Net as run_FLRW_Net_3
    from FourStep.solver_routine import run_FLRW_Net as run_FLRW_Net_4

    if STEPS_VALUE.get() != "<choose...>":
        messagebox.showinfo("Changed number of steps",
            "Cosmological constant, learning rate and clipnorm have " +
            "been set to new default values.")

    if time_steps == 1:
        DEFAULT_CC = "0.001"
        DEFAULT_LR = "0.001"
        DEFAULT_CLIPNORM = "1"
        STEPS_VALUE.set("a single timestep")
        button.config(command=lambda: run_FLRW_Net_1(root, N1, N3, NTE, LAMB, EPOCHS,
                                            LR, CLIPNORM, LOADED_WEIGHTS, INI, FIN, LOSS_THRESHOLD))
    elif time_steps == 2:
        DEFAULT_CC = "1e-10"
        DEFAULT_LR = "1e-6"
        DEFAULT_CLIPNORM = "2"
        STEPS_VALUE.set("two timesteps")
        button.config(command=lambda: run_FLRW_Net_2(root, N1, N2, N3, NTE, LAMB, EPOCHS,
                                            LR, CLIPNORM, LOADED_WEIGHTS, INI, FIN, LOSS_THRESHOLD, time_steps))
        
    elif time_steps == 3:
        DEFAULT_CC = "1e-10"
        DEFAULT_LR = "1e-5"
        DEFAULT_CLIPNORM = "1"
        STEPS_VALUE.set("three timesteps")
        button.config(command=lambda: run_FLRW_Net_3(root, N1, N2, N3, NTE, LAMB, EPOCHS,
                                            LR, CLIPNORM, LOADED_WEIGHTS, INI, FIN, LOSS_THRESHOLD, time_steps))
        
    elif time_steps == 4:
        DEFAULT_CC = "1e-10"
        DEFAULT_LR = "1e-5"
        DEFAULT_CLIPNORM = "1"
        STEPS_VALUE.set("four timesteps")
        button.config(command=lambda: run_FLRW_Net_4(root, N1, N2, N3, NTE, LAMB, EPOCHS,
                                            LR, CLIPNORM, LOADED_WEIGHTS, INI, FIN, LOSS_THRESHOLD, time_steps))
        
    UPDATED_CONFIG = True
    CHANGED_STEPS = True
    
    update_flrw_config("", 0)

def create_triangulation_menu(menubar):
    """Create a menu to choose the triangulation of the spatial hypersurfaces."""
    task_menu = tk.Menu(menubar, tearoff=0)
    task_menu.add_command(label="5-cell", command=lambda: update_flrw_config("input1", "5-cell"))
    task_menu.add_command(label="16-cell", command=lambda: update_flrw_config("input1", "16-cell"))
    task_menu.add_command(label="600-cell", command=lambda:update_flrw_config("input1","600-cell"))

    return task_menu

def create_boundary_menu(menubar):
    """Create a menu to choose the type of boundary data used: edge lengths or scale factors."""
    boundary_menu = tk.Menu(menubar, tearoff=0)
    boundary_menu.add_command(label="Edge lengths",
        command=lambda: update_flrw_config("input2", "edge lengths"))
    boundary_menu.add_command(label="Scale factors",
        command=lambda: update_flrw_config("input2", "scale factors"))

    return boundary_menu

def create_weights_menu(root_window, menubar):
    """
    Create a menu to allow the user to choose custom/pre-trained weights,
    which the network uses initially as its trainable weights.
    """
    weights_menu = tk.Menu(menubar, tearoff=0)
    weights_menu.add_command(label="Set default weights",
            command=lambda: default_weights_chosen("default"))
    weights_menu.add_command(label="Load weights",
            command=lambda: load_weights_dialog(root_window, "custom"))

    return weights_menu

def create_help_menu(root_window, menubar):
    """Create a menu containing misc features that inform and help the user."""
    help_menu = tk.Menu(menubar, tearoff=0)
    help_menu.add_command(label="Purpose", command=purpose)
    help_menu.add_command(label="Documentation", command=lambda: open_documentation())
    help_menu.add_command(label="Copyright", command=my_copyright)
    help_menu.add_command(label="Git project", command=lambda: git_message_box(root_window))

    return help_menu

def set_threshold(string):
    """Set the value for the global loss threshold."""
    global LOSS_THRESHOLD

    if string == "1e-20":
        LOSS_THRESHOLD = 1e-20
    elif string == "1e-21":
        LOSS_THRESHOLD = 1e-21
    elif string == "1e-22":
        LOSS_THRESHOLD = 1e-22
    elif string == "1e-23":
        LOSS_THRESHOLD = 1e-23
    elif string == "1e-24":
        LOSS_THRESHOLD = 1e-24
    elif string == "1e-25":
        LOSS_THRESHOLD = 1e-25
    elif string == "1e-26":
        LOSS_THRESHOLD = 1e-26
    elif string == "1e-27":
        LOSS_THRESHOLD = 1e-27
    elif string == "1e-28":
        LOSS_THRESHOLD = 1e-28

    messagebox.showinfo("Info", f"The threshold for the loss has been set to {LOSS_THRESHOLD}.")

def create_threshold_menu(menubar):
    """Create a menu to allow the user to choose different thresholds for the loss."""
    threshold_menu = tk.Menu(menubar, tearoff=0)
    threshold_menu.add_command(label="1e-20", command=lambda: set_threshold("1e-20"))
    threshold_menu.add_command(label="1e-21", command=lambda: set_threshold("1e-21"))
    threshold_menu.add_command(label="1e-22", command=lambda: set_threshold("1e-22"))
    threshold_menu.add_command(label="1e-23", command=lambda: set_threshold("1e-23"))
    threshold_menu.add_command(label="1e-24", command=lambda: set_threshold("1e-24"))
    threshold_menu.add_command(label="1e-25", command=lambda: set_threshold("1e-25"))
    threshold_menu.add_command(label="1e-26", command=lambda: set_threshold("1e-26"))
    threshold_menu.add_command(label="1e-27", command=lambda: set_threshold("1e-27"))
    threshold_menu.add_command(label="1e-28", command=lambda: set_threshold("1e-28"))

    return threshold_menu

def on_closing_root_window(root_window):
    """Set closing dialog."""
    if messagebox.askyesno("Quit", "Do you really want to quit FLRW-Net's user interface?"):

        messagebox.showinfo("Goodbye", "Hope to see you soon :)")
        root_window.destroy()

def run_user_interface():
    """Specify the UI's windows and features."""

    # Generate the root window and make it invisible
    root = tk.Tk()
    root.withdraw()

    # Bind the close window event if nothing happended.
    # Upon closing the root window ask whether FLRW-Net's UI should really be closed.
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing_root_window(root))

    # Set global parameters to adjust the FLRW_CONFIG label text properly
    global TRIANGULATION_VALUE, BOUNDARY_VALUE, WEIGHTS_VALUE, FLRW_CONFIG, STEPS_VALUE
    TRIANGULATION_VALUE = tk.StringVar()
    TRIANGULATION_VALUE.set("<choose...>")
    BOUNDARY_VALUE = tk.StringVar()
    BOUNDARY_VALUE.set("<choose...>")
    WEIGHTS_VALUE = tk.StringVar()
    WEIGHTS_VALUE.set("default")
    STEPS_VALUE = tk.StringVar()
    STEPS_VALUE.set("<choose...>")

    # Set a visible statement on the root window how FLRW-Net is currently configured
    FLRW_CONFIG = tk.Label(root, text=f"You want to compute {STEPS_VALUE.get()} for the "
        f"{TRIANGULATION_VALUE.get()} model\n and to prescribe {BOUNDARY_VALUE.get()} as"
        f" boundary data using {WEIGHTS_VALUE.get()} weights.", font=("Arial", 12), bg="white")

    # Configure the statement
    FLRW_CONFIG.grid(row=2, column=1, pady=(0, 20), sticky="nsew")

    # Simulate a loading at startup with a progress bar in a separate window.
    # Further code is executed in the meanwhile.
    startup_loading_window(root)

    def create_timesteps_menu(menubar):
        """Create a menu to choose the type of boundary data used: edge lengths or scale factors."""
        timesteps_menu = tk.Menu(menubar, tearoff=0)
        timesteps_menu.add_command(label="1 step",
            command=lambda: update_button_command(root, compute_button, 1))
        timesteps_menu.add_command(label="2 steps",
            command=lambda: update_button_command(root, compute_button, 2))
        timesteps_menu.add_command(label="3 steps",
            command=lambda: update_button_command(root, compute_button, 3))
        timesteps_menu.add_command(label="4 steps",
            command=lambda: update_button_command(root, compute_button, 4))

        return timesteps_menu
    
    def create_menubar(root_window):
        """Create a menubar for the root window."""
        menu_bar = tk.Menu(root_window)

        # Create separate menus
        triangulation_menu = create_triangulation_menu(menu_bar)
        boundary_menu = create_boundary_menu(menu_bar)
        timesteps_menu = create_timesteps_menu(menu_bar)
        weights_menu = create_weights_menu(root_window, menu_bar)
        threshold_menu = create_threshold_menu(menu_bar)
        help_menu = create_help_menu(root_window, menu_bar)

        # Add menus to the menu bar
        menu_bar.add_cascade(label="Time steps", menu=timesteps_menu)
        menu_bar.add_cascade(label="Triangulation", menu=triangulation_menu)
        menu_bar.add_cascade(label="Boundary data", menu=boundary_menu)
        menu_bar.add_cascade(label="Weights", menu=weights_menu)
        menu_bar.add_cascade(label="Loss threshold", menu=threshold_menu)
        menu_bar.add_cascade(label="Help", menu=help_menu)

        return menu_bar

    # Create a menubar for the root window
    menubar = create_menubar(root)

    # Configure the root window
    root.title("FLRW-Net")
    root.lift()
    root.config(bg="white", menu=menubar)
    root.iconbitmap("Icon.ico")
    root.resizable(False, False)

    # Center the root window on the screen
    centering_window(root, width=800, height=230)

    # Ensure that the default weights are set initially
    global LOADED_WEIGHTS
    LOADED_WEIGHTS = None

    def on_entry1_click(event, compute_button):
        """Clear the placeholder text when the cosmological constant entry widget is clicked."""
        if entry1.get() == DEFAULT_CC:
            entry1.delete(0, tk.END)
            entry1.config(fg="black")

        # Disable the compute button upon changing the parameters.
        compute_button.config(state="disabled")

    def on_focus_out1(event):
        """Restore the placeholder text when the cosmological constant entry widget loses focus."""
        if not entry1.get():
            entry1.insert(0, DEFAULT_CC)
            entry1.config(fg="gray")

    def on_entry2_click(event, compute_button):
        """Clear the placeholder text when the initial boundary entry widget is clicked."""
        if entry2.get() == DEFAULT_EDGE_INI:
            entry2.delete(0, tk.END)
            entry2.config(fg="black")

        # Disable the compute button upon changing the parameters.
        compute_button.config(state="disabled")

    def on_focus_out2(event):
        """Restore the placeholder text when the initial boundary entry widget loses focus."""
        if not entry2.get():
            entry2.insert(0, DEFAULT_EDGE_INI)
            entry2.config(fg="gray")

    def on_entry3_click(event, compute_button):
        """Clear the placeholder text when the final boundary entry widget is clicked."""
        if entry3.get() == DEFAULT_EDGE_FIN:
            entry3.delete(0, tk.END)
            entry3.config(fg="black")

        # Disable the compute button upon changing the parameters.
        compute_button.config(state="disabled")

    def on_focus_out3(event):
        """Restore the placeholder text when the final boundary entry widget loses focus."""
        if not entry3.get():
            entry3.insert(0, DEFAULT_EDGE_FIN)
            entry3.config(fg="gray")

    def on_entry4_click(event, compute_button):
        """Clear the placeholder text when the EPOCHS entry widget is clicked."""
        if entry4.get() == DEFAULT_EPOCHS:
            entry4.delete(0, tk.END)
            entry4.config(fg="black")

        # Disable the compute button upon changing the parameters.
        compute_button.config(state="disabled")

    def on_focus_out4(event):
        """Restore the placeholder text when the EPOCHS entry widget loses focus."""
        if not entry4.get():
            entry4.insert(0, DEFAULT_EPOCHS)
            entry4.config(fg="gray")

    def on_entry5_click(event, compute_button):
        """Clear the placeholder text when the learning rate entry widget is clicked."""
        if entry5.get() == DEFAULT_LR:
            entry5.delete(0, tk.END)
            entry5.config(fg="black")

        # Disable the compute button upon changing the parameters.
        compute_button.config(state="disabled")

    def on_focus_out5(event):
        """Restore the placeholder text when the learning rate entry widget loses focus."""
        if not entry5.get():
            entry5.insert(0, DEFAULT_LR)
            entry5.config(fg="gray")

    def on_entry6_click(event, compute_button):
        """Clear the placeholder text when the CLIPNORM entry widget is clicked."""
        if entry6.get() == DEFAULT_CLIPNORM:
            entry6.delete(0, tk.END)
            entry6.config(fg="black")

        # Disable the compute button upon changing the parameters.
        compute_button.config(state="disabled")

    def on_focus_out6(event):
        """Restore the placeholder text when the CLIPNORM entry widget loses focus."""
        if not entry6.get():
            entry6.insert(0, DEFAULT_CLIPNORM)
            entry6.config(fg="gray")

    def validate_entries(entry1, entry2, entry3, entry4, entry5, entry6):
        """Check whether the entry fields only contain numbers."""
        # Check the entry field for the cosmological constant
        if entry1.get() == "":
            return False
        try:
            test1 = float(entry1.get())
            if test1 < 0:
                messagebox.showwarning("Warning",
                "The value for the cosmological constant must not be negative!")
                return False
        except ValueError:
            messagebox.showwarning("Warning",
                "The value for the cosmological constant is not numeric!")
            return False

        # Check the entry field for the initial boundary
        if entry2.get() == "":
            return False
        try:
            test2 = float(entry2.get())
            if test2 < 0:
                messagebox.showwarning("Warning",
                "The value for the initial boundary must not be negative!")
                return False
        except ValueError:
            messagebox.showwarning("Warning", "The value for the initial boundary is not numeric!")
            return False

        # Check the entry field for the final boundary
        if entry3.get() == "":
            return False
        try:
            test3 = float(entry3.get())
            if test3 < 0 or test3 == test2:
                messagebox.showwarning("Warning", "The value for the final boundary "
                "must neither be negative nor must it match the value for the initial boundary!")
                return False
        except ValueError:
            messagebox.showwarning("Warning", "The value for the final boundary is not numeric!")
            return False

        # Check the entry field for the EPOCHS
        if entry4.get() == "":
            return False
        try:
            test4 = float(entry4.get())
            if test4 < 1:
                messagebox.showwarning("Warning", "You must at least train for one epoch!")
                return False
        except ValueError:
            messagebox.showwarning("Warning", "The value for the EPOCHS is not numeric!")
            return False

        # Check the entry field for the learning rate
        if entry5.get() == "":
            return False
        try:
            test5 = float(entry5.get())
            if test5 <= 0:
                messagebox.showwarning("Warning",
                "The value for the learning rate must be positive!")
                return False
        except ValueError:
            messagebox.showwarning("Warning", "The value for the learning rate is not numeric!")
            return False

        # Check the entry field for the CLIPNORM
        if entry6.get() == "":
            return False
        try:
            test6 = float(entry6.get())
            if test6 <= 0 or test6 > 2:
                messagebox.showwarning("Warning",
                "The value for the CLIPNORM must be positive and smaller than 2"
                "\n(Experimental upper limit: still subject to change.)!")
                return False
        except ValueError:
            messagebox.showwarning("Warning", "The value for the CLIPNORM is not numeric!")
            return False

        return True

    def confirm_choices(triangulation, boundary, compute_button,
                        entry1, entry2, entry3, entry4, entry5, entry6):
        """Store the entries, set up all computation params and enable the compute button."""

        # Set all model and training parameters to be global vars
        global N1, N2, N3, NTE, LAMB, EPOCHS, LR, CLIPNORM, INI, FIN

        # Take the focus from the entry fields and set it to the root window.
        root.focus()

        if validate_entries(entry1, entry2, entry3, entry4, entry5, entry6):
            # Set triangulation parameters
            N1, N2, N3, NTE,triangulation_warning=hdr.set_triangulation_params(triangulation.get())

            # Set boundary parameters
            INI, FIN, boundary_warning = hdr.set_boundary_params(boundary, entry2, entry3, N3)

            # Handle the different warnings
            LAMB, EPOCHS, LR, CLIPNORM = hdr.handle_warning_messages(triangulation_warning,
                                            boundary_warning, triangulation, compute_button,
                                            entry1, entry4, entry5, entry6, N1, N3, NTE, INI, FIN)

    def generate_entry_field_labels():
        """Create labels for the entry fields."""
        label1 = tk.Label(root, text="Cosmological constant:", background="white")
        label2 = tk.Label(root, text="Initial boundary:", background="white")
        label3 = tk.Label(root, text="Final boundary:", background="white")
        label4 = tk.Label(root, text="Epochs:", background="white")
        label5 = tk.Label(root, text="Learning rate:", background="white")
        label6 = tk.Label(root, text="Clipnorm:", background="white")

        # Arrange labels in a grid
        label1.grid(row=4, column=0, pady=0, padx=0)
        label2.grid(row=4, column=1, pady=0, padx=0)
        label3.grid(row=4, column=2, pady=0, padx=0)
        label4.grid(row=6, column=0, pady=0, padx=5)
        label5.grid(row=6, column=1, pady=0, padx=5)
        label6.grid(row=6, column=2, pady=0, padx=5)

    def generate_entry_fields(compute_button):
        """Generate the root window's entry fields."""

        # Entry field for the value of the cosmological constant
        entry1 = tk.Entry(root, fg="gray")
        entry1.insert(0, DEFAULT_CC)
        entry1.bind("<FocusIn>", lambda event: on_entry1_click(event, compute_button))
        entry1.bind("<FocusOut>", on_focus_out1)

        # Entry field for the value of the initial boundary
        entry2 = tk.Entry(root, fg="gray")
        entry2.insert(0, DEFAULT_EDGE_INI)
        entry2.bind("<FocusIn>", lambda event: on_entry2_click(event, compute_button))
        entry2.bind("<FocusOut>", on_focus_out2)

        # Entry field for the value of the final boundary
        entry3 = tk.Entry(root, fg="gray")
        entry3.insert(0, DEFAULT_EDGE_FIN)
        entry3.bind("<FocusIn>", lambda event: on_entry3_click(event, compute_button))
        entry3.bind("<FocusOut>", on_focus_out3)

        # Entry field for the value of the number of EPOCHS
        entry4 = tk.Entry(root, fg="gray")
        entry4.insert(0, DEFAULT_EPOCHS)
        entry4.bind("<FocusIn>", lambda event: on_entry4_click(event, compute_button))
        entry4.bind("<FocusOut>", on_focus_out4)

        # Entry field for the value of the learning rate
        entry5 = tk.Entry(root, fg="gray")
        entry5.insert(0, DEFAULT_LR)
        entry5.bind("<FocusIn>", lambda event: on_entry5_click(event, compute_button))
        entry5.bind("<FocusOut>", on_focus_out5)

        # Entry field for the value of the adam optimizer's CLIPNORM
        entry6 = tk.Entry(root, fg="gray")
        entry6.insert(0, DEFAULT_CLIPNORM)
        entry6.bind("<FocusIn>", lambda event: on_entry6_click(event, compute_button))
        entry6.bind("<FocusOut>", on_focus_out6)

        # Arrange entry fields in a grid
        entry1.grid(row=5, column=0, pady=10, padx=0)
        entry2.grid(row=5, column=1, pady=10, padx=0)
        entry3.grid(row=5, column=2, pady=10, padx=0)
        entry4.grid(row=7, column=0, pady=7, padx=5)
        entry5.grid(row=7, column=1, pady=7, padx=5)
        entry6.grid(row=7, column=2, pady=7, padx=5)

        return entry1, entry2, entry3, entry4, entry5, entry6

    def no_steps_chosen():
        messagebox.showerror("Error", "Number of steps not yet chosen.")

    def generate_root_buttons():
        """Generate the root window's buttons."""

        # Compute button
        compute_button = tk.Button(root, text="Compute", background="lightgray", command=lambda: no_steps_chosen())
        compute_button.grid(row=8, column=2, pady=20, padx=0)
        compute_button.config(state="disabled")

        # Confirm inputs button
        get_inputs_button = tk.Button(root, text="Confirm choices", background="lightgray",
        command=lambda: confirm_choices(TRIANGULATION_VALUE, BOUNDARY_VALUE, compute_button,
                                        entry1, entry2, entry3, entry4, entry5, entry6))
        get_inputs_button.grid(row=8, column=1, pady=20, padx=0)

        return compute_button

    def check_for_changes(entry1, entry5, entry6):
        """Disable the compoute button upon a change in the triangulation or the boundary data."""
        global UPDATED_CONFIG, CHANGED_STEPS
        if UPDATED_CONFIG:
            UPDATED_CONFIG = False

            # Disable the compute button upon changing the parameters.
            compute_button.config(state="disabled")

            if CHANGED_STEPS:
                CHANGED_STEPS = False

                entry1.delete(0, 'end')
                entry1.insert(0, DEFAULT_CC)
                entry5.delete(0, 'end')
                entry5.insert(0, DEFAULT_LR)
                entry6.delete(0, 'end')
                entry6.insert(0, DEFAULT_CLIPNORM)

        # Check every 0.2 seconds for changes
        root.after(200, lambda: check_for_changes(entry1, entry5, entry6))

    # Generate and set the buttons of the root window
    compute_button = generate_root_buttons()

    # Generate and set the entry field labels of the root window
    generate_entry_field_labels()

    # Generate and set the entry fields of the root window
    entry1, entry2, entry3, entry4, entry5, entry6 = generate_entry_fields(compute_button)

    # Force the user to save his changes before entering the next computation
    #root.after(100, check_for_changes)
    root.after(200, lambda: check_for_changes(entry1, entry5, entry6))

    # Run the Tkinter event loop
    root.mainloop()
