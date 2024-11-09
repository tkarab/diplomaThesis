import time

import numpy as np
from keras import utils
import tensorflow
import keras
import random
import os
from tqdm import tqdm

from constants import *
from helper_functions import *
from plot_functions import *
from data_augmentation import *
from preprocessing import *


"""
PARAMETERS
    - mode: "train" or "val"
    - val_day_time: (day,time) of session picked for validation where day is between 1 to 5 and time 1 or 2
    - task_type: "inter_session" or "intra_session"
"""
class TaskGeneratorEx4IntraSubject(utils.Sequence):
    def __init__(self, subject, val_day_time:tuple, way, shot, mode, task_type, network_type, preprocessing_config:dict,aug_enabled:bool, aug_config:dict=None, batch_size:int=1, batches: int = 1000, rms_win_size:int=200):
        self.way = way
        self.shot = shot
        self.mode = mode
        self.network_type = network_type

        self.subject = subject
        self.val_day_time = val_day_time
        self.task_type = task_type

        self.preproc_config = preprocessing_config
        self.aug_enabled = aug_enabled
        self.aug_config = aug_config

        self.batch_size = batch_size
        self.batches_per_epoch = batches

        self.rms = rms_win_size
        self.segment_win_size = self.get_window_size()
        self.channels = 14 #always

        # g,r domains
        self.g_domain = [1,2,3,4,5,6,7]
        self.r_domain = [1,2,3,4,5,6,7,8,9,10,11,12]

        # (day,time) domain
        if mode == "val":
            self.dt_domain = [val_day_time]
        elif mode == "train":
            self.dt_domain = [(day,time) for day in [1,2,3,4,5] for time in [1,2] if (day,time) != val_day_time]

        # (r,d,t) combinations in case of inter_session training
        if task_type == "inter_session":
            self.r_d_t_combinations = [(r,)+d_t for r in self.r_domain for d_t in self.dt_domain]

        self.data = {}
        self.data_raw = {}
        self.data_aug = {}
        self.segments = {}

        self.get__data()

        self.task_generator = None
        if network_type == "protoNet":
            self.task_generator = self.task_generator_protoNet
        elif network_type == "siamNet":
            self.task_generator = self.task_generator_siamNet

        return


    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, index):
        support_batch, query_batch, labels_batch = self.task_generator(index)
        return [support_batch, query_batch], labels_batch

    def get_window_size(self):
        return int(self.preproc_config['params']['SEGMENT']['fs']*(self.preproc_config['params']['SEGMENT']['window_size_ms']/1000))
    def get__data(self):
        # raw data
        print("Collecting Raw Data...")
        progress_bar = tqdm(total=len(self.dt_domain), desc="Augmentation", unit=" reps")
        for d,t in self.dt_domain:
            path = os.path.join(PROCESSED_DATA_PATH_DB6, f'rms{self.rms}', f'db6_s{self.subject}', f's{self.subject}_d{d}_t{t}.npz')
            with np.load(path) as data:
                for key, value in data.items():
                    key_dt = key + f"d{d:02d}t{t:02d}"
                    self.data_raw[key_dt] = np.copy(value)

            progress_bar.set_postfix(day_time=f'd{d:2d}t{t:2d}')
            progress_bar.update(1)  # Update progress bar by 1
        progress_bar.close()

        self.data, self.segments = apply_preprocessing_db6(
            data=self.data_raw,
            experiment_type="intra_subject",
            final_sample_subfix="g07r12",
            total=len(self.dt_domain),
            config_dict=self.preproc_config
        )

        if self.aug_enabled:
            self.data_aug = [apply_augmentation_db6(
                data=self.data,
                experiment_type="intra_subject",
                config_dict=self.aug_config,
                total=len(self.dt_domain),
                final_sample_subfix="g07r12"
            )]


        return

    def task_generator_protoNet(self, index):
        support_batch = np.zeros((self.batch_size, self.way, self.shot, self.segment_win_size, self.channels, 1))
        query_batch = np.zeros((self.batch_size, 1, self.segment_win_size, self.channels, 1))
        labels_batch = np.zeros((self.batch_size, self.way))

        for batch_no in range(self.batch_size):
            # Select N random gestures
            task_gestures = random.sample(self.g_domain, self.way)
            # Select 1 out of the N to be the query one and keep it index (which of the 5 it is)
            query_gesture_index, chosen_query_gest = random.choice(list(enumerate(task_gestures)))
            shot_list = [self.shot] * self.way
            shot_list[query_gesture_index] += 1

            if self.task_type == "intra_session":
                dt_choice = random.choice(self.dt_domain)
                reps = [random.sample(self.r_domain, shots) for shots in shot_list]
                r_d_t_combs = [[(gest_reps[j],)+dt_choice for j in range(len(gest_reps))] for gest_reps in reps]

            elif self.task_type == "inter_session":
                r_d_t_combs = [random.sample(self.r_d_t_combinations, shots) for shots in shot_list]

            query_rdt = r_d_t_combs[query_gesture_index].pop()

            support_set = []
            for i in range(self.way):
                gest_samples = []
                for j in range(self.shot):
                    key = getKeyWithDayTime(self.subject, task_gestures[i], *r_d_t_combs[i][j])
                    gest_samples.append(self.get_segment_of_semg(key))
                support_set.append(gest_samples)

            query_key = getKeyWithDayTime(self.subject, task_gestures[query_gesture_index], *query_rdt)
            query_gest = self.get_segment_of_semg(query_key)
            labels = utils.to_categorical([query_gesture_index], num_classes=self.way)

            support_batch[batch_no] = np.array(support_set)
            query_batch[batch_no] = np.array(query_gest)
            labels_batch[batch_no] = labels[0]

        return support_batch, query_batch, labels_batch

    def task_generator_siamNet(self, index):
        x0_batch = np.zeros((self.batch_size, self.segment_win_size, self.channels, 1))
        x1_batch = np.zeros((self.batch_size, self.segment_win_size, self.channels, 1))
        labels_batch = np.zeros((self.batch_size))

        for batch_no in range(self.batch_size):
            label = random.choice([0, 1])

            if self.task_type == "intra_session":
                dt = random.choice(self.dt_domain)
                if label == 0:
                    g0,g1 = random.sample(self.g_domain,2)
                    key0 = getKeyWithDayTime(self.subject, g0, random.choice(self.r_domain), *dt)
                    key1 = getKeyWithDayTime(self.subject, g1, random.choice(self.r_domain), *dt)
                elif label == 1:
                    g = random.choice(self.g_domain)
                    r0,r1 = random.sample(self.r_domain,2)
                    key0 = getKeyWithDayTime(self.subject, g, r0, *dt)
                    key1 = getKeyWithDayTime(self.subject, g, r1, *dt)

            elif self.task_type == "inter_session":
                if label == 0:
                    g0, g1 = random.sample(self.g_domain, 2)
                    key0 = getKeyWithDayTime(self.subject, g0, *random.choice(self.r_d_t_combinations))
                    key1 = getKeyWithDayTime(self.subject, g1, *random.choice(self.r_d_t_combinations))

                elif label == 1:
                    g = random.choice(self.g_domain)
                    rdt0, rdt1 = random.sample(self.r_d_t_combinations, 2)
                    key0 = getKeyWithDayTime(self.subject, g, *rdt0)
                    key1 = getKeyWithDayTime(self.subject, g, *rdt1)

            x0 = self.get_segment_of_semg(key0)
            x1 = self.get_segment_of_semg(key1)
            x0_batch[batch_no] = x0
            x1_batch[batch_no] = x1
            labels_batch[batch_no] = label


        return x0_batch, x1_batch, labels_batch

    def get_segment_of_semg(self, key):
        # random segment of signal
        segment_start = random.choice(self.segments[key])
        indices = np.arange(segment_start, segment_start + self.segment_win_size)

        if not self.aug_enabled:
            x = np.take(self.data[key], indices, axis=0)

        else:
            #TODO - Might be more depending on the number of unique augmentation techniques used
            ind = np.random.choice([0, 1])  # 0: non-aug, 1: aug
            # data_aug is a list containing multiple dicts, in case of many different augmentation combinations used
            x = np.take([self.data[key], random.choice(self.data_aug)[key]][ind], indices, axis=0)

        return x
