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

def getMetricChoice():
    choice = int(input("Select metric:\n 0: train_accuracy\n 1: train_loss\n 2: val_accuracy\n 3: val_loss\n\nYour choice: "))
    if choice not in [0,1,2,3]:
        return -1
    return choice

ex='1'
way=5
shot=5
# root = get_results_dir_fullpath(ex=ex, N=way,k=shot)
root = constants.RESULTS_PATH_ALL_EXPERIMENTS
subdir = "test_dense"
metric = "val_accuracy"

selected_files = select_files(root)

mode = "FORMAT_RESULTS"
# mode = "PLOT_RESULTS"

plot_mode = ""

plt.ioff()

metrics = ["train_accuracy", "train_loss", "val_accuracy", "val_loss"]

if mode == "PLOT_RESULTS":
    title = input("title: ")
    legend_title = input("legend title: ")
    plot_mode = input("Select mode\n\t0: Experiment vs Experiment (multiple files)\n\t1: Same Experiment (one file)\n")
    if plot_mode == '0':
        while True:
            choice = getMetricChoice()
            if choice != -1:
                break


for filepath in selected_files:
    if mode == "FORMAT_RESULTS":
        if not filepath.endswith("res.txt"):
            print("Wrong filename, must end with 'res.txt'")
            continue

        fullpath_out = filepath.replace('.txt', 'ults.txt')
        parse_training_results_to_txt(input_file=filepath, output_file=fullpath_out)
        os.remove(filepath)

    elif mode == "PLOT_RESULTS":
        if plot_mode == "0":
            metric = metrics[choice]
            label = input(f"label for file '{filepath.split('/')[-1]}': ")
            plot_train_results_exp_vs_exp(input_file=filepath, label=label, title=title, legend_title=legend_title, metric=metric)

        else:
            plot_train_results_same_exp(input_file=filepath, title=title, legend_title="", metric="accuracy",input_type="filename")

print()