import os.path

from scipy.io import loadmat
import numpy as np
import constants
import time
from matplotlib import pyplot as plt

from helper_functions import *
from plot_functions import *


subjects = [1,2,3,4,5,6,7,8,9,10]
days = [1,2,3,4,5]
times = [1,2]
total_reps = [1,2,3,4,5,6,7,8,9,10,11,12]
channels = [0,1,2,3,4,5,6,7,10,11,12,13,14,15]

data_needed = ['emg', 'restimulus', 'rerepetition']
signal_dict = {}

root = r'C:\Users\ΤΑΣΟΣ\Desktop\Σχολή\Διπλωματική\Δεδομένα\Ninapro\DB6'

sub = 7
print(f"subject {sub}")

start_time = time.time()

for sub in [2]:
    for day in [2]:
        for t in [2]:
            print(f'\nD: {day}, T:{t}')

            signal_dict = {}
            gesture_number = 0
            filename = rf'DB6_s{sub}\S{sub}_D{day}_T{t}.mat'
            dt_filepath = os.path.join(root, filename)
            data = loadmat(dt_filepath)
            emg = data['emg']
            restimulus = data['restimulus']
            rerepetition = data['rerepetition']

            L = len(restimulus)

            previous_gesture = 0
            current_signal = np.array([]).astype(np.float32)

            for i in range(L):
                current_gesture = restimulus[i]

                if current_gesture != 0:
                    if previous_gesture == 0:
                        gesture_number += (rerepetition[i].item() == 1)
                        current_signal = np.append(current_signal, emg[i,channels])
                    else:
                        current_signal = np.vstack((current_signal, emg[i,channels]))

                elif (current_gesture == 0 and previous_gesture != 0) or (i == L - 1 and current_gesture != 0):
                    key = getKey(sub, gesture_number, rerepetition[i - 1].item())
                    print(key, f"({time.time() - start_time:.2f}s)")
                    start_time = time.time()
                    signal_dict[key] = np.copy(current_signal)
                    current_signal = np.array([]).astype(np.float32)

                previous_gesture = current_gesture

            path = constants.SEPARATED_DATA_PATH + rf'\db6\db6_s{sub}\s{sub}_d{day}_t{t}.npz'
            np.savez(path, **signal_dict)
            os.remove(dt_filepath)

        print()



# print("--- {:.3f} seconds ---".format((time.time() - start_time)))


