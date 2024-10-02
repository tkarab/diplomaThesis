import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, ifft

def plot_sEMG(emg, fs, channels=[1,2,3,4,5,6], title='', figure_name = 'Figure', export=False, filename=''):
  fig = plt.figure()
  fig.canvas.manager.set_window_title(figure_name)

  time = np.arange(len(emg))
  time = time/fs
  N = len(channels)

  # max = np.max(np.abs(emg[:,channels]))
  for i in range(N):
    emg_channel = emg[:,channels[i]-1]
    max = np.max(np.abs(emg_channel))
    emg_norm = emg_channel/max
    offset = 2.5*(N-i-1)
    plt.plot(time,emg_norm+offset, label='channel {}'.format(i+1))

  lgd = plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
  plt.xlabel('time')
  plt.ylabel('channels')
  plt.title(title)
  plt.show()

"""
    DESCRIPTION
    Plots the spectrum of a 1-d channel of an semg signal
    If the all the channels of the signal are inserted it keeps the first one
    It shows the spectrum Amplitude for frequencies up to fs/2, either in multiples of pi or in Hz
        
    PARAMETERS
    x   : the semg signal channel
    fs  : sampling frequency
    title : plot title
    figure_name : name for the figure that will be initialized
    
    RETURNS
    Nothing
"""
def plotSpectrum(x,fs,showInMultiplesOfPi=False, title='Φάσμα emg σήματος', figure_name='Spectrum of sEMG Channel'):
  fig = plt.figure()
  fig.canvas.manager.set_window_title(figure_name)

  if(len(x.shape)>1):
    x = x[:,0]

  X = fft(x,5*len(x))
  L = len(X)
  L_lim = L//2
  if showInMultiplesOfPi:
    freq_lim = 1
    x_text = "Digital Frequency in multiples of pi"
  else:
    freq_lim = fs/2
    x_text = "Analog Frequency in Hz ([0,Fs/2])"

  df = freq_lim/(L_lim)

  freq = np.arange(0,freq_lim,df)
  magnitude = np.abs(X)[:L_lim]
  plt.plot(freq, magnitude,color='r',label='sEMG Spectrum')

  plt.xlabel(x_text)
  plt.title(title)
  plt.show()

  return

"""
DECRIPTION
  For plotting the values of a dictionary using a bar
  If keys have multiple values (same number of values for each key) they will be displayed using multiple colors (same for each category)

PARAMETERS
  data_dict : dictionary whose values are either lists (of same nuber of elements) or 1 value
  i.e. data_dict = {
        'a': [3, 4, 3],
        'b': [5, 7, 3],
        'c': [2, 2, 3]
      }
"""
def plotDictBar(data_dict:dict):
  keys = list(data_dict.keys())[:100]
  values = np.array(list(data_dict.values()))[:100]

  # Number of categories
  num_categories = values.shape[1]

  # Colors for each category
  colors = plt.cm.viridis(np.linspace(0, 1, num_categories))

  # Create a stacked bar chart
  fig, ax = plt.subplots()

  # Plot each category
  for i in range(num_categories):
    if i == 0:
      ax.bar(keys, values[:, i], color=colors[i], label=f'non-augmented')
    else:
      ax.bar(keys, values[:, i], bottom=np.sum(values[:, :i], axis=1), color=colors[i], label=f'augmented')

  plt.xlabel('Key')
  plt.ylabel('Value')

  # Add a legend
  plt.legend()

  # Show the plot
  plt.show()

  return

