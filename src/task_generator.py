import numpy as np
from keras import utils
import tensorflow
import keras
import random




class TaskGenerator(utils.Sequence):
    def __init__(self, experiment:str, way:int, shot:int, mode:str,channels:int=12, window_size:int=40, batches: int = 1000):
        self.experiment = experiment
        self.way = way
        self.shot = shot
        self.mode = mode
        self.channels = channels
        self.window_size = window_size
        self.getData()
        self.batches = batches


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
        path = r'C:\Users\ΤΑΣΟΣ\Desktop\Σχολή\Διπλωματική\Δεδομένα\processed\db2\concat\db2_ex{}_{}.npz'.format(self.experiment[0], self.mode)
        self.data = np.load(path)

    def getKeys(self,*entries:tuple) -> list:
        return ['s{}g{}r{}'.format(s,g,r) for s,r,g in entries]

    def getKeys_in_order(self,*entries:tuple) -> list:
        return ['s{}g{}r{}'.format(s,g,r) for s,g,r in entries]

    def get_s_r_pairs(self) -> list:
        return [(s,r) for s in self.s_domain for r in self.r_domain]

    def __getitem__(self, item):
        suport, query, label = self.task_generator()  # activates either generate_task_1() or generate_task_2a() depending on the experiment
        print()
        print(label)
        return [suport, query], label
    def __len__(self):
        return self.batches

    def generate_task_1(self):
        key_list = []
        support_set = np.empty((self.way, self.shot, self.channels, self.window_size, 1))
        query_image = np.empty((1, self.channels, self.window_size, 1))

        task_gestures = random.sample(self.g_domain, self.way)
        query_gesture_index, chosen_query_gest = random.choice(list(enumerate(task_gestures)))
        shot_list = [self.shot]*self.way
        shot_list[query_gesture_index] += 1

        support_pairs = [random.sample(self.s_r_pairs, shot_number) for shot_number in shot_list]
        for i,g in enumerate(task_gestures):
            sgr_list = [pair + (g,) for pair in support_pairs[i]]
            key_list.append(self.getKeys(*sgr_list))

        query_key = key_list[query_gesture_index].pop()

        for i,gest_key_list in enumerate(key_list):
            support_set[i] = np.array([random.choice(self.data[key]) for key in gest_key_list])

        query_image = np.array([random.choice(self.data[query_key])])*np.ones(self.way)[:, np.newaxis, np.newaxis, np.newaxis]
        label = utils.to_categorical(query_gesture_index, num_classes=self.way)
        #printKeys(key_list)
        return support_set, query_image, label

    def generate_task_2a(self):
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
            support_set[i] = np.array([random.choice(self.data[key]) for key in gest_key_list])

        query_image = np.array([random.choice(self.data[query_key])])*np.ones(self.way)[:, np.newaxis, np.newaxis, np.newaxis]
        label = utils.to_categorical([query_gesture_index], num_classes=self.way)

        # printKeys(key_list)

        return support_set, query_image, label

def printKeys(keys):
    print()
    for j in range(len(keys[0])):
        print("| |", end='')
        for i in range(len(keys)):
            print("{:<8}".format(keys[i][j]),end="| |")
        print()


