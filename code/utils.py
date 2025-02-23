import scipy 
import csv
import os
import numpy as np


import pyedflib


def readEDF(filename, ns,
            root_dir = '',
           req_chns = ['FP1','FP2','F7','F3','FZ','F4','F8','T3','C3','CZ','C4','T4','T5','P3','PZ','P4','T6','O1','O2'], 
           ):
    
    f = pyedflib.EdfReader(root_dir+filename)
    n = f.signals_in_file
    print(n)
    data =  {key: None for key in req_chns}
    fs = {key: None for key in req_chns}
    for i in range(n):
        
        chn = str(f.signal_label(i)).split(" ")[1].split("-")[0]
        print(chn)
        if chn in req_chns:
           
            data[chn] = f.readSignal(i)                
            fs[chn] = f.getSampleFrequencies()[i]

    f.close()
    for chn in req_chns:
        if fs[chn] == None:                
                data[chn] = np.zeros(ns)
                fs[chn] = fs['FP1']
      
    data = np.array(list(data.values()))
    fs = np.array(list(fs.values()))
    #print(data.shape, fs)
    return data, fs


def read_manifest(filename, d=';'):
   
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=d)
        dicts = list(reader)
    return dicts

def write_manifest(new_manifest, fname='untitled_mn.csv', d=';'):
    with open(fname, 'w') as csvfile:
        fieldnames = new_manifest[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')

        writer.writeheader()
        for i in range(len(new_manifest)):
            writer.writerow(new_manifest[i] )

def resample_eeg(x, fs, ns, fsnew=200.):
    
    return sig.resample(x, int(np.float(ns)*fsnew/fs))

def applyLowPass(x, fs, fc=30, N=4):
    """Apply a low-pass filter to the signal
    """
    wc = fc / (fs / 2)
    b, a = scipy.signal.butter(N, wc)
    return scipy.signal.filtfilt(b, a, x, method='gust', axis=-1)


def applyHighPass(x, fs, fc=1.6, N=4):
    """Apply a high-pass filter to the signal
    """
    wc = fc / (fs / 2)
    b, a = scipy.signal.butter(N, wc, btype='highpass')
    return scipy.signal.filtfilt(b, a, x, method='gust', axis=-1)

def clip(x, clip_level):
    """Clip the signal to a given standard deviation"""
    mean = np.mean(x, axis=-1).reshape(-1, 1)
    std = np.std(x, axis=-1).reshape(-1, 1)
    return np.clip(x, mean - clip_level * std, mean + clip_level * std)

def preprocess(x, fs, f2=30, f1=1.6, N=4, clip_level=2, axis = -1):
    x_fil = applyHighPass(applyLowPass(x, fs, f2, N), fs, f1, N)
    return clip(x_fil, clip_level)
