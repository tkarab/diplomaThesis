import os

dataPath = r'C:\Users\ΤΑΣΟΣ\Desktop\Σχολή\Διπλωματική\Δεδομένα'
ninaproPath = r'C:\Users\ΤΑΣΟΣ\Desktop\Σχολή\Διπλωματική\Δεδομένα\Ninapro'
separatedDataPath = dataPath + r'\Separated'

def getDB2Path():
    return os.path.join(ninaproPath,'DB2')
def getDB5Path():
    return os.path.join(ninaproPath,'DB5')

def getRaw_DB_Subject_path(db, s):
    db_string = f'DB{db}'
    return os.path.join(ninaproPath, db_string,'_'.join([db_string,f's{s}']))

def getRaw_subject_exercise_file_path(s,e):
    return f'S{s}_E{e}_A1.mat'