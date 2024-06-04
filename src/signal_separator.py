from scipy.io import loadmat
import numpy as np
import constants
import time

def getSubjectPath(db,subject,exercise):
    return constants.ninaproPath + r'\DB{}\DB{}_s{}\S{}_E{}_A1.mat'.format(db,db,subject,subject,exercise)

def getSubjectKey(subject,gesture,repetition):
    return r's{}g{}r{}'.format(subject,gesture,repetition)

cols = ['subject','exercise','repetition','emg_signal','samples']

data_needed = ['emg','restimulus','rerepetition']
signal_dict = {}
database = 2
total_subjects = 40
total_exercises = 49

#start_time = time.time()
for sub in range(1,total_subjects+1):
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
                name = getSubjectKey(sub,gesture_number,rerepetition[i-1].item())
                signal_dict[name] = np.transpose(current_signal)
                current_signal = np.array([]).astype(np.float32)    
                                
            previous_gesture = current_gesture
            
#path = constants.separatedDataPath+'\DB'+str(database)+'\db'+str(database)+'s'+str(sub)+'_sep.npz'
#np.savez(path,**signal_dict)
    
    

#print("--- {:.3f} seconds ---".format((time.time() - start_time)))

        
