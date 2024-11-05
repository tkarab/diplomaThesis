from scipy.io import loadmat
import numpy as np
import constants
import time
from matplotlib import pyplot as plt

from helper_functions import *


def getSubjectPath(db,subject,exercise):
    return constants.NINAPRO_PATH + r'\DB{}\DB{}_s{}\S{}_E{}_A1.mat'.format(db, db, subject, subject, exercise)


db_sgr = {
    2: {"total_exercises":3, "total_gestures": 49, "total_subjects":40, "total_reps":6},
    4: {"total_exercises":3, "total_gestures": 52, "total_subjects":10, "total_reps":6},
    5: {"total_exercises":3, "total_gestures": 52, "total_subjects":10, "total_reps":6}
}

cols = ['subject','exercise','repetition','emg_signal','samples']

data_needed = ['emg','restimulus','rerepetition']
signal_dict = {}
database = 4
total_subjects = db_sgr[database]["total_subjects"]
total_exercises = db_sgr[database]["total_exercises"]

start_time = time.time()
for sub in range(1,total_subjects+1):
    print(f"subject {sub}")
    signal_dict = {}
    gesture_number = 0
    
    for exer in range(1,total_exercises+1):

        data = loadmat(getSubjectPath(db=database,subject=sub,exercise=exer), variable_names = data_needed)
        emg = data['emg']
        restimulus = data['restimulus']
        rerepetition = data['rerepetition']

        L = len(restimulus)

        previous_gesture = 0
        current_signal = np.array([]).astype(np.float32)
        
        for i in range(L):
            current_gesture = restimulus[i]
            
            if current_gesture!=0:
                if previous_gesture == 0:
                    gesture_number += (rerepetition[i].item() == 1)
                    current_signal = np.append(current_signal,emg[i])
                else:
                    current_signal = np.vstack((current_signal,emg[i]))
                    
            elif (current_gesture==0 and previous_gesture!=0) or (i==L-1 and current_gesture!=0):
                name = getKey(sub,gesture_number,rerepetition[i-1].item())
                print(name, f"({time.time()-start_time:.2f}s)")
                start_time = time.time()
                signal_dict[name] = np.transpose(current_signal)
                current_signal = np.array([]).astype(np.float32)
                                
            previous_gesture = current_gesture
    print()
            
path = constants.SEPARATED_DATA_PATH+'\db'+str(database)+'.npz'
np.savez(path,**signal_dict)
    
    

#print("--- {:.3f} seconds ---".format((time.time() - start_time)))

        
