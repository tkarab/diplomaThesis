import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, ifft

def plot_sEMG(emg, fs, channels=[1,2,3,4,5,6], title='', figure_name = 'Figure', export=False, filename=''):
  fig = plt.figure()
  fig.canvas.manager.set_window_title(figure_name)

  time = np.arange(len(emg))
  time = time/fs
  N = len(channels)

  for i in range(N):
    emg_channel = emg[:,channels[i]-1]
    max = np.max(np.abs(emg_channel))
    emg_norm = emg_channel/max
    offset = 2.5*(N-i-1)
    plt.plot(time,emg_norm+offset, label='channel {}'.format(i+1))

  lgd = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
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