import numpy as np
import os
import sys

def slicelist(li, W = 8, s = 3):
    L = len(li) - (W-s)
    (m,d) = divmod(L,s)
    print(m,d)
    windowed = [li[i:i+W] for i in range(0,m*s,s)]
    if d!=0:
        windowed.append(li[-W:])
    return windowed

def apply_sliding_window(arr:np.ndarray, W , s ):
    arr = np.reshape(arr,arr.shape+(1,))
    L = arr.shape[1] - (W-s)
    shape = (arr.shape[0], W, 1)
    (m, d) = divmod(L, s)
    slices = np.array([arr[:,i:i+W] for i in range(0,m*s,s)])
    if d!=0:
        slices = np.append(slices, arr[:,-W:].reshape((1,) + shape), axis=0)

    return slices

def getfilename(s):
    return 'db2_s{}_processed.npz'.format(s)

def get_new_file_number(*filenames : str):
    namelist = []
    for name in filenames:
        namelist.append(name.split('_')[1][1:])
    return '-'.join(namelist)


# L = 47
# li = [i for i in range(L)]
#
# # sliced = slicelist(li,s=2)
# a = np.arange(22).reshape(2,-1,1)
# sliding_window(a)

path = r'C:\Users\ΤΑΣΟΣ\Desktop\Σχολή\Διπλωματική\Δεδομένα\processed\db2'

s1 = getfilename('1-2')
s2 = getfilename('3-4')
print(getfilename(get_new_file_number(s1, s2)))

# data = np.load(r'C:\Users\ΤΑΣΟΣ\Desktop\Σχολή\Διπλωματική\Δεδομένα\emg.npz')
# data_sliced = sliding_window(data['arr_0'], W=40, s=10)
# for i in range(1,41):
#     fullname = path+ "\\" + getfilename(i)
#     data = np.load(fullname)
#
#     for key in list(data.keys()):
#         new_data = data[key]

path = path+'\concat'
directories = os.listdir(path)
if 'properties.txt' in directories:
    directories.remove('properties.txt')

for i in range(0,len(directories),2):
    name1 = directories[i]
    name2 = directories[i+1]
    fullname1 = path+ "\\" + name1
    fullname2 = path+ "\\" + name2
    data1 = np.load(fullname1)
    data2 = np.load(fullname2)
    new_data = {**dict(data1),**dict(data2)}
    new_name = getfilename(get_new_file_number(name1,name2))
    np.savez(path + "\\"+ new_name, **new_data)
    # os.remove(fullname1)
    # os.remove(fullname2)
    print("merged {}, {} into {}\n\n".format(name1, name2, new_name))
#
# for name in directories:
#     os.remove(path + "\\"+name)

print()