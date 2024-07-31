import time
import numpy as np
from keras import utils
import tensorflow
import keras
import random
import os
import csv
import pandas as pd

from constants import *
from helper_functions import *
from plot_functions import *
from data_augmentation import *
from preprocessing import *


class FileInfoProvider:
    def __init__(self, db, rms, N, k, ex, mode):
        if db == 2:
            self.data_directory = os.path.join(PROCESSED_DATA_PATH_DB2, get_rmsRect_dirname(db, rms))
            self.data_filepath = os.path.join(self.data_directory, get_rmsRect_dirname(db, rms)+'.npz')
        elif db == 1:
            self.data_directory = PROCESSED_DATA_PATH_DB1
            self.data_filepath = os.path.join(self.data_directory,'db1_raw.npz')
        elif db == 5:
            self.data_directory = PROCESSED_DATA_PATH_DB5
            self.data_filepath = os.path.join(self.data_directory,'db5_raw.npz')

        self.N = N
        self.k = k
        self.ex = ex
        self.mode = mode

    def getDataDirectoryPath(self):
        return self.data_directory

    def getDataFullPath(self):
        return self.data_filepath

    def getTasksFileFullPath(self):
        return os.path.join(TASKS_FILE_PATH_DB2,get_tasks_filename(ex=self.ex, N=self.N, k=self.k, mode=self.mode))


"""
DESCRIPTION

PARAMETERS
    - database: 1,2 or 5
    - data_intake: either "csv" (to read from a csv) or "generate" (to generate tasks)
    

"""
class TaskGenerator(utils.Sequence):
    def __init__(self, experiment:str, way:int, shot:int, mode:str,data_intake:str, database:int ,  preprocessing_config:dict,aug_enabled:bool, aug_config:dict, batch_size:int=1, batches: int = 1000, rms_win_size:int=200, print_labels = False, print_labels_frequency = 100):
        self.experiment = experiment
        self.way = way
        self.shot = shot
        self.mode = mode
        self.data_intake = data_intake

        self.db = database

        self.preproc_config = preprocessing_config
        self.window_size = self.getWindowSize()
        self.rms_win_size = rms_win_size
        self.fileInfoProvider = FileInfoProvider(self.db, self.rms_win_size, N = self.way, k = self.shot, ex = self.experiment, mode=self.mode)

        self.aug_enabled = aug_enabled
        if self.aug_enabled == True:
            self.aug_config = aug_config
            self.data_aug = {}

        # Only used if data_intake == 'csv'
        self.support_keys = []
        self.support_seg_start = []
        self.query_keys = []
        self.query_seg_start = []
        self.query_gest_indices = []

        self.get__data()
        self.keyAppDict = self.get_key_app_dict()
        self.channels = self.getNumberOfChannels()

        self.batch_size = batch_size
        self.batches_per_epoch = batches
        self.print_labels = print_labels
        self.print_label_freq = print_labels_frequency

        if self.experiment == '1':
            self.s_domain = list(range(1,41))
            self.g_domain = list(range(1,50))
            self.r_domain = {'train':[1,3,4,6], 'test':[2,5]}[self.mode]
            self.s_r_pairs = self.get_s_r_pairs()

        elif self.experiment in ['2a', '2b']:
            self.s_domain = {'train': list(range(1,28)), 'val': list(range(28,33)),'test': list(range(33,41))}[self.mode]
            self.g_domain = list(range(1,50))
            self.r_domain = list(range(1,7))

            if experiment == '2b':
                self.s_r_pairs = self.get_s_r_pairs()

        elif self.experiment == '3':
            self.s_domain = list(range(1, 41))
            self.g_domain = {'train': list(range(1,35)), 'val': list(range(35,41)), 'test': list(range(41,50))}[self.mode]
            self.r_domain = list(range(1, 7))
            self.s_r_pairs = self.get_s_r_pairs()

        if self.data_intake == "csv":
            # t1 = time.time()
            self.load_tasks_from_file()
            # print(f"total time for loading tasks : {time.time()-t1:.2f}")
            self.task_generator = self.get_premade_keys
        else:
            if self.experiment == '2a':
                self.task_generator = self.generate_task_keys_2a
            elif self.experiment in ['1','2b','3']:
                self.task_generator = self.generate_task_keys

        return

    def __getitem__(self, index):
        all_keys = self.task_generator(index)
        support_batch, query_batch, labels_batch = self.get_task_data_based_on_keys(*all_keys)

        return [support_batch, query_batch], labels_batch

    def __len__(self):
        return self.batches_per_epoch

    def get__data(self):
        self.data, self.segments = apply_preprocessing(self.fileInfoProvider.getDataFullPath(), self.preproc_config, self.db)

        if self.aug_enabled:
            self.data_aug = apply_augmentation(self.data, self.aug_config)

    def load_tasks_from_file(self):
        full_path = self.fileInfoProvider.getTasksFileFullPath()
        print('Loading tasks...')

        df = pd.read_csv(full_path)
        s_keys = df.iloc[:,:-3:2]
        s_seg = df.iloc[:,1:-3:2]
        q_keys = df.iloc[:,-3]
        q_seg = df.iloc[:,-2]
        q_label = df.iloc[:,-1]

        self.support_keys = np.reshape(s_keys.to_numpy(),[len(s_seg),self.way,self.shot])
        self.support_seg_start = np.reshape(s_seg.to_numpy(dtype=np.int32),[len(s_seg),self.way,self.shot])

        self.query_keys = q_keys.to_numpy()
        self.query_seg_start = q_seg.to_numpy()
        self.query_gest_indices = q_label.to_numpy()

        print("\n...tasks have been loaded.")

    def reshape_support_set(self,support_flattened):
        support_set = []
        for i in range(self.way):
            support_set.append(support_flattened[i*self.shot:(i+1)*self.shot])

        return support_set

    def get_key_app_dict(self):
        keyAppDict = {}
        for key in self.data.keys():
            if self.aug_enabled:
                keyAppDict[key] = [0,0]
            else:
                keyAppDict[key] = 0
        return keyAppDict


    def getNumberOfChannels(self):
        random_key = random.choice(list(self.data.keys()))
        random_sample = self.data[random_key]
        return random_sample.shape[1]

    def getWindowSize(self):
        w_ms = self.preproc_config['params']['SEGMENT']['window_size_ms']
        fs = self.preproc_config['params']['SEGMENT']['fs']
        return int((w_ms * fs) / 1000)

    def getKeys(self,*entries:tuple) -> list:
        return [getKey(s,g,r) for s,r,g in entries]

    def getKeys_in_order(self,*entries:tuple) -> list:
        return [getKey(s,g,r) for s,g,r in entries]

    def get_s_r_pairs(self) -> list:
        return [(s,r) for s in self.s_domain for r in self.r_domain]

    def plotKeyAppHist(self):
        plotDictBar(self.keyAppDict)


    def get_segment_of_semg(self, key, segment_start):
        # segment_start = random.choice(self.segments[key])
        indices = np.arange(segment_start, segment_start+self.window_size)
        if not self.aug_enabled:
            x = np.take(self.data[key],indices,axis=0)
            self.keyAppDict[key] += 1
        else:
            ind = np.random.choice([0,1]) # 0: non-aug, 1: aug
            x = np.take([self.data[key], self.data_aug[key]][ind],indices,axis=0)
            self.keyAppDict[key][ind] += 1

        return x

    def generate_task_keys(self, index):
        support_set_keys = []
        query_keys = []
        query_gest_indices = []
        support_segments_starting_indices = []
        query_segments_starting_indices = []

        for i in range(self.batch_size):
            support_set = []
            start_indices = []

            task_gestures = random.sample(self.g_domain, self.way)
            query_gesture_index, chosen_query_gest = random.choice(list(enumerate(task_gestures)))
            shot_list = [self.shot]*self.way
            shot_list[query_gesture_index] += 1

            support_pairs = [random.sample(self.s_r_pairs, shot_number) for shot_number in shot_list]
            for i,g in enumerate(task_gestures):
                sgr_list = [pair + (g,) for pair in support_pairs[i]]
                category_keys = self.getKeys(*sgr_list)
                support_set.append(category_keys)
                start_indices.append([random.choice(self.segments[key]) for key in category_keys])

            query_key = support_set[query_gesture_index].pop()
            query_seg_start_index = start_indices[query_gesture_index].pop()

            support_set_keys.append(support_set)
            query_keys.append(query_key)
            query_gest_indices.append(query_gesture_index)
            support_segments_starting_indices.append(start_indices)
            query_segments_starting_indices.append(query_seg_start_index)

        return support_set_keys, query_keys, query_gest_indices, support_segments_starting_indices, query_segments_starting_indices

    def generate_task_keys_2a(self, index):
        support_set_keys = []
        query_keys = []
        query_gest_indices = []

        for i in range(self.batch_size):
            support_set = []

            task_gestures = random.sample(self.g_domain, self.way)
            query_gesture_index, chosen_query_gest = random.choice(list(enumerate(task_gestures)))
            chosen_subject = random.choice(self.s_domain)
            shot_list = [self.shot] * self.way
            shot_list[query_gesture_index] += 1

            reps = [random.sample(self.r_domain, shot_number) for shot_number in shot_list]
            for i,g in enumerate(task_gestures):
                sgr_list = [(chosen_subject,rep,g) for rep in reps[i]]
                support_set.append(self.getKeys(*sgr_list))

            query_key = support_set[query_gesture_index].pop()

            support_set_keys.append(support_set)
            query_keys.append(query_key)
            query_gest_indices.append(query_gesture_index)

        return support_set_keys, query_keys, query_gest_indices

    def get_premade_keys(self,index):
        ind = np.arange(index*self.batch_size, (index+1)*self.batch_size)
        support_keys = self.support_keys[ind]
        query_keys = self.query_keys[ind]
        support_seg = self.support_seg_start[ind]
        query_seg = self.query_seg_start[ind]
        gest_ind = self.query_gest_indices[ind]

        return support_keys, query_keys, gest_ind, support_seg, query_seg

    def get_task_data_based_on_keys(self, support_keys, query_keys, query_gesture_indices, support_seg_starting_indices, query_seg_starting_indices):
        support_set_batch = []
        query_batch = []
        labels_batch = []

        # Each list should have length == self.batch_size
        for i in range(len(support_keys)):

            support_set = np.empty((self.way, self.shot, self.window_size, self.channels, 1))
            query_image = np.empty((1, self.channels, self.window_size, 1))

            for j, gest_key_list in enumerate(support_keys[i]):
                shots = [self.get_segment_of_semg(key,support_seg_starting_indices[i][j][k]) for k,key in enumerate(gest_key_list)]
                support_set[j] = np.array(shots)

            query_image = np.expand_dims(self.get_segment_of_semg(query_keys[i],query_seg_starting_indices[i]), axis=0)
            label = utils.to_categorical([query_gesture_indices[i]], num_classes=self.way)

            support_set_batch.append(support_set)
            query_batch.append(query_image)
            labels_batch.append(label[0])

        return np.array(support_set_batch), np.array(query_batch), np.array(labels_batch)



def create_csv_lines(support_keys,query_keys,query_gest_ind,support_seg_start,query_seg_start):
    batch_size,way,shot = np.array(support_keys).shape
    task_lines = np.empty([batch_size,2*(way*shot+1)+1],dtype='U9')
    s_keys_np = np.reshape(np.array(support_keys),[batch_size,way*shot])
    q_keys_np = np.expand_dims(np.array(query_keys),-1)
    s_q_keys = np.concatenate((s_keys_np,q_keys_np),axis=1)
    s_seg_np = np.reshape(np.array(support_seg_start),[batch_size,way*shot])
    q_seg_np = np.expand_dims(np.array(query_seg_start),-1)
    s_q_seg = np.concatenate((s_seg_np,q_seg_np),axis=1)

    task_lines[:,:-1:2] = s_q_keys
    task_lines[:,1:-1:2] = s_q_seg
    task_lines[:,-1] = query_gest_ind

    return task_lines

"""
PARAMETERS
    - mode : 'train', 'test' or 'val'
"""
def save_tasks(task_lines,db,experiment,way,shot,mode):
    with open(os.path.join(f'tasks/DB{db}',get_tasks_filename(ex=experiment,N=way,k=shot,mode=mode)),'w', newline='') as file:
        writer = csv.writer(file)
        for task in task_lines:
            writer.writerow(task)

if __name__ == '__main__':
    way = 5
    shot = 5
    ex = '1'
    db = 2
    num_batches = 10000
    batch_size = 64
    mode = 'train'
    preproc_config = get_config_from_json_file('preproc', 'db2_no_lpf')
    Gen = TaskGenerator(experiment='1', way=way, shot=shot, mode=mode, data_intake='generate',database=db, preprocessing_config=preproc_config, aug_enabled=False, aug_config=None, rms_win_size=200, batch_size=batch_size, batches=num_batches, print_labels=False, print_labels_frequency=0)

    t1 = time.time()
    task_lines = np.empty([1 + num_batches*batch_size,2*(way*shot+1)+1],dtype='U9')
    task_lines[0] = [el for i in range(way) for j in range(shot) for el in (f"class{i+1}_ex{j+1}_key", f"class{i+1}_ex{j+1}_seg")] + ['query_key','query_seg','label']

    print("Preparing tasks...")
    for i in range(num_batches):
        keys = Gen.generate_task_keys(i)
        task_lines[1 + i*batch_size : 1 + (i+1)*batch_size] = create_csv_lines(*keys)
        if i%100 == 0:
            print(f"Batch {i}/{num_batches} : {time.time()-t1:.2f}")

    print(f"Batch {i+1}/{num_batches} : {time.time() - t1:.2f}")

    print("\n...tasks have been prepared")
    print(f"total time for {num_batches*batch_size} ({way}way-{shot}shot) tasks: {time.time()-t1:.2f}")
    save_tasks(task_lines,db=db, experiment=ex, way=way, shot=shot, mode=mode)
    print()