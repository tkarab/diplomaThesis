import os

ROOT = r'C:\Users\ΤΑΣΟΣ\Desktop\Σχολή\Διπλωματική'

DATA_PATH = ROOT + r'\Δεδομένα'
NINAPRO_PATH = ROOT + r'\Δεδομένα\Ninapro'
SEPARATED_DATA_PATH = os.path.join(DATA_PATH ,'Separated')

PROCESSED_DATA_PATH_DB1 = ROOT + r'\Δεδομένα\processed\db1'
PROCESSED_DATA_PATH_DB2 = ROOT + r'\Δεδομένα\processed\db2'
PROCESSED_DATA_PATH_DB5 = ROOT + r'\Δεδομένα\processed\db5'

RMS_DATA_PATH_DB1 = os.path.join(PROCESSED_DATA_PATH_DB1,'rms')
RMS_DATA_PATH_DB2 = os.path.join(PROCESSED_DATA_PATH_DB2,'rms')
RMS_DATA_PATH_DB5 = os.path.join(PROCESSED_DATA_PATH_DB5,'rms')

RESULTS_PATH_EX1  = ROOT + r'\Δεδομένα\Results\Experiment 1'
RESULTS_PATH_EX2A = ROOT + r'\Δεδομένα\Results\Experiment 2a'
RESULTS_PATH_EX2B = ROOT + r'\Δεδομένα\Results\Experiment 2b'
RESULTS_PATH_EX3  = ROOT + r'\Δεδομένα\Results\Experiment 3'
RESULTS_DIRECTORIES_DICT = {
    '1' : RESULTS_PATH_EX1,
    '2a': RESULTS_PATH_EX2A,
    '2b': RESULTS_PATH_EX2B,
    '3' : RESULTS_PATH_EX3
}

TASKS_FILES_PATH = ROOT + r'\Δεδομένα\tasks'
TASKS_FILE_PATH_DB1 = os.path.join(TASKS_FILES_PATH,'DB1')
TASKS_FILE_PATH_DB2 = os.path.join(TASKS_FILES_PATH,'DB2')
TASKS_FILE_PATH_DB5 = os.path.join(TASKS_FILES_PATH,'DB5')


DATA_CONFIG_PATH = r'C:\PycharmProjects\diplomaThesis\src\config'
DATA_CONFIG_PATH_PREPROC = os.path.join(DATA_CONFIG_PATH,'preproc')
DATA_CONFIG_PATH_AUG = os.path.join(DATA_CONFIG_PATH, 'aug')


