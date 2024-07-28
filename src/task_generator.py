import numpy as np
from keras import utils
import tensorflow
import keras
import random
import constants
import os
import helper_functions as hlp
from plot_functions import *
import preprocessing
import data_augmentation as aug


class DataFileInfoProvider:
    def __init__(self, db, rms):
        if db == 2:
            self.data_directory = os.path.join(constants.PROCESSED_DATA_PATH_DB2, hlp.get_rmsRect_dirname(db, rms))
            self.data_filepath = os.path.join(self.data_directory, hlp.get_rmsRect_dirname(db, rms)+'.npz')
        elif db == 1:
            self.data_directory = constants.PROCESSED_DATA_PATH_DB1
            self.data_filepath = os.path.join(self.data_directory,'db1_raw.npz')
        elif db == 5:
            self.data_directory = constants.PROCESSED_DATA_PATH_DB5
            self.data_filepath = os.path.join(self.data_directory,'db5_raw.npz')

    def getDirectoryPath(self):
        return self.data_directory

    def getDataFullPath(self):
        return self.data_filepath


"""
DESCRIPTION

PARAMETERS
    database: 1,2 or 5
    

"""
class TaskGenerator(utils.Sequence):
    def __init__(self, experiment:str, way:int, shot:int, mode:str, database:int ,  preprocessing_config:dict,aug_enabled:bool, aug_config:dict, batch_size:int=1, batches: int = 1000, rms_win_size:int=200, print_labels = False, print_labels_frequency = 100):
        self.experiment = experiment
        self.way = way
        self.shot = shot
        self.mode = mode
        self.db = database

        self.preproc_config = preprocessing_config
        self.window_size = self.getWindowSize()
        self.rms_win_size = rms_win_size
        self.dataFileInfoProvider = DataFileInfoProvider(self.db, self.rms_win_size)

        self.aug_enabled = aug_enabled
        if self.aug_enabled == True:
            self.aug_config = aug_config
            self.data_aug = {}

        self.get__data()
        self.keyAppDict = self.get_key_app_dict()
        self.channels = self.getNumberOfChannels()

        self.batch_size = batch_size
        self.batches = batches
        self.print_labels = print_labels
        self.print_label_freq = print_labels_frequency


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

            if experiment == '2a':
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

        return

    def get__data(self):
        self.data, self.segments = preprocessing.apply_preprocessing(self.dataFileInfoProvider.getDataFullPath(), self.preproc_config)

        if self.aug_enabled:
            self.data_aug = aug.apply_augmentation(self.data, self.aug_config)

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
        return [hlp.getKey(s,g,r) for s,r,g in entries]

    def getKeys_in_order(self,*entries:tuple) -> list:
        return [hlp.getKey(s,g,r) for s,g,r in entries]

    def get_s_r_pairs(self) -> list:
        return [(s,r) for s in self.s_domain for r in self.r_domain]

    def plotKeyAppHist(self):
        plotDictBar(self.keyAppDict)

    def __getitem__(self, index):
        all_keys = self.task_generator()
        support_batch, query_batch, labels_batch = self.get_task_data_based_on_keys(*all_keys)

        return [support_batch, query_batch], labels_batch

        # return [np.array(support_array), np.array(query_array)], np.array(labels_array)
    def __len__(self):
        return self.batches

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

    def generate_task_keys(self):
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

    def generate_task_keys_2a(self):
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
    def __init__(self, way, shot, ex, batch_size, num_batches):
        self.way = way
        self.shot = shot
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

            if experiment == '2a':
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

    def generate_task_keys_2a(self):
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

    def get_s_r_pairs(self) -> list:
        return [(s,r) for s in self.s_domain for r in self.r_domain]
