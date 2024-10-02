import numpy as np
# import pandas as pd
import os
import sys
import random
# import fsl_functions
# import custom_models
import constants
import time
from helper_functions import *
from plot_functions import *
from preprocessing import *
from matplotlib import pyplot as plt


data_path = r'C:\Users\ΤΑΣΟΣ\Desktop\Σχολή\Διπλωματική\Δεδομένα\processed\db2\db2_processed.npz'
mu1 = 2048
mu2 = 256
mu3 = 32

x = np.linspace(-1,1,1000)
y1 = muLaw_transform(x,mu1)
y2 = muLaw_transform(x,mu2)
y3 = muLaw_transform(x,mu3)
plt.plot(x,y1, label=f"μ={mu1}")
plt.plot(x,y2, label=f"μ={mu2}")
plt.plot(x,y3, label=f"μ={mu3}")

plt.legend()

print()


