import numpy as np
import scipy
import os
import pandas as pd
import torch
from baselines import txlstm_szpool
import pyedflib

class EDFReader:
    def __init__(self, filename, root_dir = '', req_chns = ['FP1','FP2','F7','F3','FZ','F4','F8','T3','C3','CZ','C4','T4','T5','P3','PZ','P4','T6','O1','O2']):
        self.filename = filename
        self.root_dir = root_dir
        self.req_chns = req_chns
        self.data = {key: None for key in req_chns}
        self.req_chns = set(req_chns)

        # get datetime

    
    def read(self):
        print(self.root_dir+self.filename)
        f = pyedflib.EdfReader(self.root_dir+self.filename)
        time = f.getStartdatetime()
        n = f.signals_in_file
        for i in range(n):
            chn = str(f.signal_label(i)).split('-')[0].split('\'')[1].upper()

            if chn in self.req_chns:
                self.data[chn] = f.readSignal(i)                
        f.close()
        self.data = np.array(list(self.data.values()))
        return self.data, time

class Preprocess:
    def __init__(self):
        pass
    def applyLowPass(self, x, fs, fc=30, N=4):
        """Apply a low-pass filter to the signal
        """
        wc = fc / (fs / 2)
        b, a = scipy.signal.butter(N, wc)
        return scipy.signal.filtfilt(b, a, x, method='gust', axis=-1)
    def applyHighPass(self, x, fs, fc=1.6, N=4):
        """Apply a high-pass filter to the signal
        """
        wc = fc / (fs / 2)
        b, a = scipy.signal.butter(N, wc, btype='highpass')
        return scipy.signal.filtfilt(b, a, x, method='gust', axis=-1)

    def clip(self, x, clip_level):
        """Clip the signal to a given standard deviation"""
        mean = np.mean(x, axis=-1).reshape(-1, 1)
        std = np.std(x, axis=-1).reshape(-1, 1)
        return np.clip(x, mean - clip_level * std, mean + clip_level * std)

    def preprocess(self, x, fs, f2=30, f1=1.6, N=4, clip_level=2, axis = -1):
        """Preprocess the signal by calling applyLowPass, applyHighPass, and clip"""
        x_fil = self.applyHighPass(self.applyLowPass(x, fs, f2, N), fs, f1, N)
        return self.clip(x_fil, clip_level)

    def crop(self, data):
        duration = data.shape[0]
        window = 600  # 10 min
        overlap = 120  # 2 min
        stride = window - overlap # 480s or 8 min
        leftover = (duration - window) % (window - overlap) # mod by 480
        
        return_data = []
        
        for i in range(0, duration, stride):
            end = min(i + window, duration)
            segment = data[i:end]
            
            # padding
            x = np.shape(segment)[0] 
            if end == duration and leftover > 0:
                    segment = self.pad_segment(segment)
            

            return_data.append(segment)
            if end == duration:
                break
            
            
        return_data = np.array(return_data)
        return return_data, x # returning length of last segment
    
    
    def pad_segment(self, data):
        
        data = np.array(data)
        duration = data.shape[0]
        if duration >= 600:
            return data
            
        pad_length = 600 - duration
        padding = np.zeros((pad_length, 19, 256))
        
        padded_data = np.concatenate([data, padding], axis=0).tolist()
        
        return padded_data
    
    


def vis(model, filelist, device):
        '''
        filelist is just a single .edf recording worth of 10-min segments (and leftover)
        in 2D (num_segments, 600) np array shape
        '''


        all_predictions = torch.tensor([], device=device)


        filelist = torch.tensor(filelist, device=device).float()
        
        for i, data in enumerate(filelist):
                
                inputs = data.to(device)
                inputs = inputs.unsqueeze(0)
                

                with torch.no_grad():
                        k_pred, _, _, _  = model(inputs)
                        pred_labels = torch.softmax(k_pred.squeeze(), dim=1).float()
                        

                p = pred_labels
                if i == 0:
                    all_predictions = p
                else:
                    overlap_prev = all_predictions[-120:]
                    overlap_curr = p[:120]
                    max_overlap = torch.maximum(overlap_prev, overlap_curr)
                    all_predictions = torch.cat((all_predictions[:-120], max_overlap, p[120:]))
        
        all_predictions = all_predictions.to('cpu').numpy()
        return all_predictions




class MovingAverage():
    def __init__(self, winlen=5):
        self.winlen = winlen
        self.kernel = np.ones(winlen) / winlen  # Uniform averaging kernel
    
    def avg(self, x):
        return np.convolve(x, self.kernel, mode='same')


    
def main(edf_file, outFile):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    sampling_rate = 256
    model_path = os.path.join(os.path.dirname(__file__), 'oldmodel.pth.tar')
    model_1_path = os.path.join(os.path.dirname(__file__), 'deepsoz_fold4.pth_4.tar')
    model_1 = txlstm_szpool() 
    model_1 = model_1.to(device)
    statedict = torch.load(model_1_path, map_location=device, weights_only=True)
    model_1.load_state_dict(statedict)
    model_1.eval()

    
                

    win = 25
    thresh = 0.65

    reader = EDFReader(edf_file)
    data, timevar = reader.read()
    
    processor = Preprocess()
    duration = int(data.shape[1] / sampling_rate)
    data = processor.preprocess(data, sampling_rate).reshape(data.shape[0], duration, sampling_rate).transpose(1,0,2)
    return_data, _ = processor.crop(data)

    predictions_1 = vis(model_1, return_data, device)
    smoother = MovingAverage(winlen=win)
    # experiment with 20-30 window length
    predictions_1 = (smoother.avg(predictions_1[:, 1]) > thresh) # smoothing positive class predicted probs, converting to binary
    predictions = predictions_1[:int(duration)]

    


    # we can begin BIDS conversion process at this point and return into designated directory structure

    # find seizure start times and lengths in the predictions file
    seizure_start = []
    seizure_length = []
    i = 0
    while i < len(predictions):
        if predictions[i] == 1:
            seizure_start.append(i)
            seizure_length.append(0)
            while i < len(predictions) and predictions[i] == 1:
                seizure_length[-1] += 1
                i += 1
        else:
            i += 1
    
    timevar = timevar.strftime("%Y-%m-%d %H:%M:%S")
    # timevar = 'n/a'
    
    ret = pd.DataFrame()
    if not seizure_start:
        ret = pd.DataFrame([{
        'onset': 0.0,
        'duration': float(duration),
        'eventType': 'bckg',
        'confidence': 1.0,
        'channels': 'n/a',
        'dateTime': timevar,
        'recordingDuration': float(duration)
    }])
    else:
        ret = pd.DataFrame([{
        'onset': float(seizure_start[i]),
        'duration': float(seizure_length[i]),
        'eventType': 'sz',
        'confidence': 1.0,
        'channels': 'n/a',
        'dateTime': timevar,
        'recordingDuration': float(duration)
    } for i in range(len(seizure_start))])
    
    ret.to_csv(outFile, sep='\t', index=False)
    





