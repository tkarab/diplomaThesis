import numpy as np
from keras import utils
import tensorflow
import keras
import random
import constants
import os
import helper_functions




class TaskGenerator(utils.Sequence):
    def __init__(self, experiment:str, way:int, shot:int, mode:str,channels:int=12, window_size:int=15, batches: int = 1000, print_labels = False, print_lebels_frequency = 100):
        self.experiment = experiment
        self.way = way
        self.shot = shot
        self.mode = mode
        self.channels = channels
        self.window_size = window_size
        self.getData()
        self.batches = batches
        self.print_labels = print_labels
        self.print_label_freq = print_lebels_frequency


        if self.experiment == '1':
            self.s_domain = list(range(1,41))
            self.g_domain = list(range(1,50))
            self.r_domain = {'train':[1,3,4,6], 'test':[2,5]}[self.mode]
            self.task_generator = self.generate_task_1
            self.s_r_pairs = self.get_s_r_pairs()

        elif self.experiment in ['2a', '2b']:
            self.s_domain = {'train': list(range(1,28)), 'val': list(range(28,33)),'test': list(range(33,41))}[self.mode]
            self.g_domain = list(range(1,50))
            self.r_domain = list(range(1,7))

            if experiment == '2a':
                self.task_generator = self.generate_task_2a
            else:
                self.task_generator = self.generate_task_1
                self.s_r_pairs = self.get_s_r_pairs()

        elif self.experiment == '3':
            self.s_domain = list(range(1, 41))
            self.g_domain = {'train': list(range(1,35)), 'val': list(range(35,41)), 'test': list(range(41,50))}[self.mode]
            self.r_domain = list(range(1, 7))
            self.task_generator = self.generate_task_1
            self.s_r_pairs = self.get_s_r_pairs()

        return

    def getData(self, path = r"C:\Users\ΤΑΣΟΣ\Desktop\Σχολή\Διπλωματική\Δεδομένα\processed\db2\concat\db2_processed.npz"):
        # path = r'C:\Users\ΤΑΣΟΣ\Desktop\Σχολή\Διπλωματική\Δεδομένα\processed\db2\concat\db2_ex{}_{}.npz'.format(self.experiment[0], self.mode)
        path = os.path.join(constants.PROCESSED_DATA_PATH_DB2,'db2_processed.npz')
        seg_path = os.path.join(constants.PROCESSED_DATA_PATH_DB2,'db2_segments.npz')
        self.data = np.load(path)
        self.segments = np.load(seg_path)

    def getKeys(self,*entries:tuple) -> list:
        return ['s{}g{}r{}'.format(s,g,r) for s,r,g in entries]

    def getKeys_in_order(self,*entries:tuple) -> list:
        return ['s{}g{}r{}'.format(s,g,r) for s,g,r in entries]

    def get_s_r_pairs(self) -> list:
        return [(s,r) for s in self.s_domain for r in self.r_domain]

    def __getitem__(self, index):
        suport, query, label = self.task_generator(index)  # activates either generate_task_1() or generate_task_2a() depending on the experiment
        # print(f"~ __getitem__ ~ -> ({item})\n\n")
        # print(label)
        return [suport, query], label
    def __len__(self):
        return self.batches

    def get_segment_of_semg(self, key):
        segment_start = random.choice(self.segments[key])
        indices = np.arange(segment_start, segment_start+self.window_size)
        x = np.take(self.data[key],indices,axis=0)
        # x = np.expand_dims(x,axis=-1)
        return x

    def generate_task_1(self, index):
        key_list = []
        support_set = np.empty((self.way, self.shot, self.window_size, self.channels, 1))
        query_image = np.empty((1, self.window_size, self.channels, 1))

        task_gestures = random.sample(self.g_domain, self.way)
        query_gesture_index, chosen_query_gest = random.choice(list(enumerate(task_gestures)))
        shot_list = [self.shot]*self.way
        shot_list[query_gesture_index] += 1

        support_pairs = [random.sample(self.s_r_pairs, shot_number) for shot_number in shot_list]
        for i,g in enumerate(task_gestures):
            sgr_list = [pair + (g,) for pair in support_pairs[i]]
            key_list.append(self.getKeys(*sgr_list))

        query_key = key_list[query_gesture_index].pop()

        # Nxk support set is made up of N sets of k images from each category
        # gest_key_list contains all the keys for the i-th category
        # i.e. if i = 2nd category with gesture number g=5, for k=3 examples per category then gest_key_list would be
        #  ['s1g5r2', 's3g5r3', 's12g5r4']
        # Each key corresponds to a Lx12x15x1 segmented array (in segments of 15 samples) where L is the number of segments for each movement (could vary depending o movement length)
        # For each movement one segment is chosen randomly
        for i,gest_key_list in enumerate(key_list):
            shots = [self.get_segment_of_semg(key) for key in gest_key_list]
            support_set[i] = np.array(shots)


        query_image = np.expand_dims(self.get_segment_of_semg(query_key),axis=0)
        label = utils.to_categorical(query_gesture_index, num_classes=self.way)

        # if self.print_labels and index%self.print_label_freq == 0:
        #     print(f"index={index}\n\n")
        #     printKeys(key_list)

        return support_set, query_image, label

    def generate_task_2a(self, index):
        key_list = []
        support_set = np.empty((self.way, self.shot, self.channels, self.window_size, 1))
        query_image = np.empty((1, self.channels, self.window_size, 1))

        task_gestures = random.sample(self.g_domain, self.way)
        query_gesture_index, chosen_query_gest = random.choice(list(enumerate(task_gestures)))
        chosen_subject = random.choice(self.s_domain)
        shot_list = [self.shot] * self.way
        shot_list[query_gesture_index] += 1

        reps = [random.sample(self.r_domain, shot_number) for shot_number in shot_list]
        for i,g in enumerate(task_gestures):
            sgr_list = [(chosen_subject,rep,g) for rep in reps[i]]
            key_list.append(self.getKeys(*sgr_list))

        query_key = key_list[query_gesture_index].pop()

        for i, gest_key_list in enumerate(key_list):
            shots = [self.get_segment_of_semg(key) for key in gest_key_list]
            support_set[i] = np.array(shots)

        query_image = np.repeat(np.expand_dims(self.get_segment_of_semg(query_key),axis=0),self.way,axis=0)
        label = utils.to_categorical([query_gesture_index], num_classes=self.way)

        # if self.print_labels and index%self.print_label_freq == 0:
        #     print(f"index={index}\n\n")
        #     printKeys(key_list)


        return support_set, query_image, label



