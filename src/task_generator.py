import time

import numpy as np
from keras import utils
import tensorflow
import keras
import random
import os
import preprocessing
import csv
from constants import *
from helper_functions import *
from plot_functions import *
from data_augmentation import *


class FileInfoProvider:
    def __init__(self, db, rms, N, k, ex):
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

    def getDataDirectoryPath(self):
        return self.data_directory

    def getDataFullPath(self):
        return self.data_filepath

    def getTasksFileFullPath(self):
        return os.path.join(TASKS_FILE_PATH_DB2,get_tasks_filename(ex=self.ex, N=self.N, k=self.k))


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
        self.fileInfoProvider = FileInfoProvider(self.db, self.rms_win_size, N = self.way, k = self.shot, ex = self.experiment)

        self.aug_enabled = aug_enabled
        if self.aug_enabled == True:
            self.aug_config = aug_config
            self.data_aug = {}

        # Only used if data_intake == 'csv'
        self.support_keys = []
        self.query_keys = []
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
            self.load_tasks_from_file()
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

        # return [np.array(support_array), np.array(query_array)], np.array(labels_array)

    def __len__(self):
        return self.batches_per_epoch

    def get__data(self):
        self.data, self.segments = preprocessing.apply_preprocessing(self.fileInfoProvider.getDataFullPath(), self.preproc_config)

        if self.aug_enabled:
            self.data_aug = apply_augmentation(self.data, self.aug_config)

    def load_tasks_from_file(self):
        full_path = self.fileInfoProvider.getTasksFileFullPath()
        print('Loading tasks...')
        with open(full_path,'r',newline='') as file:
            t1 = time.time()
            rows = sum(1 for line in csv.reader(file))
            self.support_keys = np.empty([rows, self.way, self.shot],dtype='U9')
            self.query_keys = np.empty([rows],dtype='U9')
            self.query_gest_indices = np.empty([rows],dtype='i4')
            file.seek(0)
            reader = csv.reader(file)

            for j,task_line in enumerate(reader):
                query_gest_index = int(task_line.pop())
                query_key = task_line.pop()
                support_set_keys = self.reshape_support_set(task_line)

                self.support_keys[j] = support_set_keys
                self.query_keys[j] = query_key
                self.query_gest_indices[j] = query_gest_index

                if(j%10000 == 0):
                    print(f"{j}/{rows} : {time.time()-t1:.2f}")

        print("...tasks have been loaded.")

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


    def get_segment_of_semg(self, key):
        segment_start = random.choice(self.segments[key])
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

        for i in range(self.batch_size):
            support_set = []
            task_gestures = random.sample(self.g_domain, self.way)
            query_gesture_index, chosen_query_gest = random.choice(list(enumerate(task_gestures)))
            shot_list = [self.shot]*self.way
            shot_list[query_gesture_index] += 1

            support_pairs = [random.sample(self.s_r_pairs, shot_number) for shot_number in shot_list]
            for i,g in enumerate(task_gestures):
                sgr_list = [pair + (g,) for pair in support_pairs[i]]
                support_set.append(self.getKeys(*sgr_list))

            query_key = support_set[query_gesture_index].pop()

            support_set_keys.append(support_set)
            query_keys.append(query_key)
            query_gest_indices.append(query_gesture_index)

        return support_set_keys, query_keys, query_gest_indices

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
        support = self.support_keys[ind]
        query = self.query_keys[ind]
        gest_ind = self.query_gest_indices[ind]
        return support, query, gest_ind

    def get_task_data_based_on_keys(self, support_keys, query_keys, query_gesture_indices):
        support_set_batch = []
        query_batch = []
        labels_batch = []

        # Each list should have length == self.batch_size
        for i in range(len(support_keys)):

            support_set = np.empty((self.way, self.shot, self.window_size, self.channels, 1))
            query_image = np.empty((1, self.channels, self.window_size, 1))

            for j, gest_key_list in enumerate(support_keys[i]):
                shots = [self.get_segment_of_semg(key) for key in gest_key_list]
                support_set[j] = np.array(shots)

            query_image = np.expand_dims(self.get_segment_of_semg(query_keys[i]), axis=0)
            label = utils.to_categorical([query_gesture_indices[i]], num_classes=self.way)

            support_set_batch.append(support_set)
            query_batch.append(query_image)
            labels_batch.append(label[0])

        return np.array(support_set_batch), np.array(query_batch), np.array(labels_batch)


"""
DESCRIPTION
    Generates tasks, using only their keys. Stores these tasks in csv format.
    These data are N lists of k keys(+1 query key) which are reshaped into 1 Nxk+1 long list
    
"""
class TaskGeneratorKeysOnly:
    def __init__(self, way, shot, mode:str, ex, batch_size, num_batches):
        self.path = f'tasks/DB{2}'

        self.way = way
        self.shot = shot
        self.mode = mode
        self.experiment = ex
        self.batch_size = batch_size
        self.num_batches = num_batches

        if self.experiment == '1':
            self.s_domain = list(range(1,41))
            self.g_domain = list(range(1,50))
            self.r_domain = {'train':[1,3,4,6], 'test':[2,5]}[self.mode]
            self.task_generator = self.generate_task_keys
            self.s_r_pairs = self.get_s_r_pairs()

        elif self.experiment in ['2a', '2b']:
            self.s_domain = {'train': list(range(1,28)), 'val': list(range(28,33)),'test': list(range(33,41))}[self.mode]
            self.g_domain = list(range(1,50))
            self.r_domain = list(range(1,7))

            if self.experiment == '2a':
                self.task_generator = self.generate_task_keys_2a
            else:
                self.task_generator = self.generate_task_keys
                self.s_r_pairs = self.get_s_r_pairs()

        elif self.experiment == '3':
            self.s_domain = list(range(1, 41))
            self.g_domain = {'train': list(range(1,35)), 'val': list(range(35,41)), 'test': list(range(41,50))}[self.mode]
            self.r_domain = list(range(1, 7))
            self.task_generator = self.generate_task_keys
            self.s_r_pairs = self.get_s_r_pairs()

    def generate_task_keys(self):
        task_lines = []

        for i in range(self.batch_size*self.num_batches):
            support_set = []
            task_gestures = random.sample(self.g_domain, self.way)
            query_gesture_index, chosen_query_gest = random.choice(list(enumerate(task_gestures)))
            shot_list = [self.shot]*self.way
            shot_list[query_gesture_index] += 1

            support_pairs = [random.sample(self.s_r_pairs, shot_number) for shot_number in shot_list]
            for i,g in enumerate(task_gestures):
                sgr_list = [pair + (g,) for pair in support_pairs[i]]
                support_set_class = self.getKeys(*sgr_list)
                if g == chosen_query_gest:
                    query_key = support_set_class.pop()
                support_set += support_set_class

            support_set.append(query_key)
            support_set.append(query_gesture_index)

            task_lines.append(support_set)


        return task_lines

    def generate_task_keys_2a(self):
        task_lines = []

        for i in range(self.batch_size*self.num_batches):
            support_set = []

            task_gestures = random.sample(self.g_domain, self.way)
            query_gesture_index, chosen_query_gest = random.choice(list(enumerate(task_gestures)))
            chosen_subject = random.choice(self.s_domain)
            shot_list = [self.shot] * self.way
            shot_list[query_gesture_index] += 1

            reps = [random.sample(self.r_domain, shot_number) for shot_number in shot_list]
            for i,g in enumerate(task_gestures):
                sgr_list = [(chosen_subject,rep,g) for rep in reps[i]]
                support_set_class = self.getKeys(*sgr_list)
                if g == chosen_query_gest:
                    query_key = support_set_class.pop()
                support_set += support_set_class

            support_set.append(query_key)
            support_set.append(query_gesture_index)

            task_lines.append(support_set)

        return task_lines

    def generate_tasks(self):
        self.task_lines = self.task_generator()

        return

    def save_tasks(self):
        with open(os.path.join(self.path,f'ex{self.experiment}{self.way}way{self.shot}shot.csv'),'w', newline='') as file:
            writer = csv.writer(file)
            for task in self.task_lines:
                writer.writerow(task)


    def getKeys(self,*entries:tuple) -> list:
        return [getKey(s,g,r) for s,r,g in entries]

    def get_s_r_pairs(self) -> list:
        return [(s,r) for s in self.s_domain for r in self.r_domain]


if __name__ == '__main__':
    way = 5
    shot = 5
    ex = '1'
    num_batches = 10000
    batch_size = 64
    gen = TaskGeneratorKeysOnly(way=way,shot=shot, mode='train', ex=ex, num_batches=num_batches, batch_size=batch_size)
    t1 = time.time()
    gen.generate_tasks()
    gen.save_tasks()

    print(f"total time for {num_batches*batch_size} ({way}way-{shot}shot) tasks: {time.time()-t1:.2f}")