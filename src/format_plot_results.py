import constants
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog

from helper_functions import *
from plot_functions import *
from preprocessing import *

def select_files(initial_dir=""):
    # Create a root window (it won't show up)
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open file dialog and allow multiple selection
    if not initial_dir=="":
        file_paths = filedialog.askopenfilenames(title="Select files", initialdir=initial_dir)
    else:
        file_paths = filedialog.askopenfilenames(title="Select files")

    # Convert to a list of file paths
    file_paths = list(file_paths)

    # Optionally print or return file paths
    for file in file_paths:
        print(file)

    return file_paths


ex='1'
way=5
shot=5
root = get_results_dir_fullpath(ex=ex, N=way,k=shot)
subdir = "test_dense"
metric = "val_accuracy"

selected_files = select_files(root)

mode = "FORMAT_RESULTS"
# mode = "PLOT_RESULTS"

plt.ioff()

for filepath in selected_files:
    if mode == "FORMAT_RESULTS":
        fullpath_out = filepath.replace('.txt', 'ults.txt')
        parse_training_results_to_txt(input_file=filepath, output_file=fullpath_out)
        os.remove(filepath)

    elif mode == "PLOT_RESULTS":
        label = input(f"label for file '{filepath.split('/')[-1]}': ")
        plot_train_results(input_file=filepath, label=label, metric="val_accuracy")

print()