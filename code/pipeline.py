import numpy as np
import scipy
import pandas as pd
import os
import torch
from torch import nn
# import szcore_evaluation
from baselines import txlstm_szpool
import pyedflib
import matplotlib.pyplot as plt
from timescoring import scoring, visualization
from timescoring.annotations import Annotation
import argparse

class EDFReader:
    def __init__(self, filename, ns, root_dir = '', req_chns = ['FP1','FP2','F7','F3','FZ','F4','F8','T3','C3','CZ','C4','T4','T5','P3','PZ','P4','T6','O1','O2']):
        self.filename = filename
        self.ns = ns
        self.root_dir = root_dir
        self.req_chns = req_chns
        self.data = {key: None for key in req_chns}
        self.fs = {key: None for key in req_chns}
        self.req_chns = set(req_chns)
    
    def read(self):
        f = pyedflib.EdfReader(self.root_dir+self.filename)
        n = f.signals_in_file
        for i in range(n):
            chn = str(f.signal_label(i)).split('-')[0].split('\'')[1].upper()
            if chn in self.req_chns:
                self.data[chn] = f.readSignal(i)                
                self.fs[chn] = f.getSampleFrequencies()[i]
        f.close()
        for chn in self.req_chns:
            if self.fs[chn] == None:                
                self.data[chn] = np.zeros(self.ns)
                self.fs[chn] = self.fs['FP1']
        self.data = np.array(list(self.data.values()))
        self.fs = np.array(list(self.fs.values()))
        return self.data, self.fs

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
            
            # for _, event in tsv_data.iterrows():
            #     if event['eventType'] == 'bckg':
            #         continue
                    
            #     sz_start = event['onset']
            #     sz_end = sz_start + event['duration']
                
            #     # Check if seizure overlaps with current segment
            #     if sz_end > i and sz_start < end:
            #         # overlap
            #         overlap_start = max(sz_start, i)
            #         overlap_end = min(sz_end, end)
                    
            #         # Adjust times 
            #         new_onset = max(0, overlap_start - i)
            #         new_duration = overlap_end - overlap_start
                    
            #         seg_tsv.append({
            #             'onset': new_onset,
            #             'duration': new_duration,
            #             'eventType': event['eventType'],
            #             'confidence': event['confidence'],
            #             'channels': event['channels'],
            #             'dateTime': event['dateTime'],
            #             'recordingDuration': float(window)
            #         })
            
            # if len(seg_tsv) > 0:
            #     seg_tsv = pd.DataFrame(seg_tsv)
            # else:
            #     # Add background event if no seizures
            #     seg_tsv = pd.DataFrame([{
            #         'onset': 0.0,
            #         'duration': float(window),
            #         'eventType': 'bckg',
            #         'confidence': 1.0,
            #         'channels': 'n/a',
            #         'dateTime': tsv_data['dateTime'].iloc[0],
            #         'recordingDuration': float(window)
            #     }])
            
            # padding
            x = np.shape(segment)[0] 
            if end == duration and leftover > 0:
                    segment = self.pad_segment(segment)
            
            
            # np.save(data_dir + f"/{seg_name}.npy", segment)
            # seg_tsv.to_csv(tsv_dir + f"/{seg_name}.tsv", sep='\t', index=False)
            return_data.append(segment)
            if end == duration:
                break
            
            
        return_data = np.array(return_data)
        return return_data, x # returning length of last segment
    
    # def label_and_save(self, save_path, segment_name):
    
    #     label_dir = save_path + 'labels'
    #     labels = np.zeros(600)
        

    #     tsv_data = pd.read_csv(f"{save_path}annotations/{segment_name}.tsv", sep='\t')
    #     for _, row in tsv_data.iterrows():
    #         if row['eventType'] != 'bckg':
    #             start = int(row['onset'])
    #             end = int((row['onset'] + row['duration']))
    #             if start < len(labels):
    #                 labels[start:min(end, len(labels))] = 1 # pad
                    
    #     np.save(label_dir + f"/{segment_name}_labels.npy", labels)
    
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


    
def pipeline(win: int, thresh: float) -> None:






    sampling_rate = 256
    subjects = [
        "sub-078",
        "sub-17",
        "sub-00",
        "sub-11",
        "sub-10",
        "sub-177",
        "sub-330",
        "sub-167",
        "sub-314",
        "sub-064"
    ]

    



    # subjects = os.listdir(root)
    # subjects = list(filter(lambda x: x.startswith('sub-'), subjects))


        # % TESTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
        # % TESTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model_1_path = f'/projectnb/seizuredet/Sz-challenge/retrain/baselines/txlstm/outer_4/new_retrained_model.pth_4.tar'
    model_1 = txlstm_szpool() 
    model_1 = model_1.to(device)
    statedict = torch.load(model_1_path, map_location=device, weights_only=True)
    model_1.load_state_dict(statedict)
    model_1.eval()


    model_2_path = f'/projectnb/seizuredet/Sz-challenge/retrain/baselines/txlstm/outer_5/new_retrained_model.pth_5.tar'
    model_2 = txlstm_szpool() 
    model_2 = model_2.to(device)
    statedict = torch.load(model_2_path, map_location=device, weights_only=True)
    model_2.load_state_dict(statedict)
    model_2.eval()


    overall_sensitivity = []
    overall_precision = []
    overall_f1 = []
    overall_fpRate = []



    # remove later
    manifest = pd.read_csv('/projectnb/seizuredet/Sz-challenge/code/manifest.csv')
    

    for subfol in subjects:
        mnlist = [mnitem for  _, mnitem in manifest.iterrows() if mnitem['subject'] == subfol]
        dataset = mnlist[0]['dataset']
        root = f'/projectnb/seizuredet/Sz-challenge/{dataset}/unprocessed/'
        # save = f'/projectnb/seizuredet/Sz-challenge/testing/processed/'
        output = f'/projectnb/seizuredet/Sz-challenge/output/ensemble-{win}-{thresh}/'
        # os.makedirs(save, exist_ok=True)
        os.makedirs(output, exist_ok=True)



        subject_sensitivity = []
        subject_precision = []
        subject_f1 = []
        subject_fpRate = []

        for sess in os.listdir(root+subfol):
            files = os.listdir(root+subfol+'/'+sess+'/eeg/')
            os.makedirs(output+subfol+'/'+sess+'/eeg/', exist_ok=True)

            edffiles = list(filter(lambda x: x.endswith('.edf'), files))
            tsvfiles = list(filter(lambda x: x.endswith('.tsv'), files))
            edffiles = sorted(edffiles)
            tsvfiles = sorted(tsvfiles)
            for curredf, currtsv in zip(edffiles, tsvfiles):
                df=pd.read_csv(root+subfol+'/'+sess+'/eeg/'+currtsv,sep='\t')
                duration = df['recordingDuration'][0]
                datetime = df['dateTime'][0]
                if duration < 600:
                    continue


                
                ns = int(duration * sampling_rate)

                reader = EDFReader(curredf.split("eeg/")[0], ns, root+subfol+'/'+sess+'/eeg/')
                data, _ = reader.read()
                processor = Preprocess()
                data = processor.preprocess(data, 256)
                assert duration == data.shape[1] / sampling_rate, f"Duration mismatch: {duration} != {data.shape[1] / sampling_rate}"
                
                data = data.reshape(data.shape[0], int(data.shape[1] / sampling_rate), sampling_rate).transpose(1,0,2)
                print(data)
                return_data, _ = processor.crop(data)
                # for f in files:
                #     processor.label_and_save(save, f)

                # run inference on each of these files
                predictions_1 = vis(model_1, return_data, device)
                smoother = MovingAverage(winlen=win)
                # experiment with 20-30 window length
                predictions_1 = (smoother.avg(predictions_1[:, 1]) > thresh) # smoothing positive class predicted probs, converting to binary
                predictions_1 = predictions_1[:int(duration)]

                predictions_2 = vis(model_2, return_data, device)
                smoother = MovingAverage(winlen=win)
                # experiment with 20-30 window length
                predictions_2 = (smoother.avg(predictions_2[:, 1]) > thresh) # smoothing positive class predicted probs, converting to binary
                predictions_2 = predictions_2[:int(duration)]

                predictions = np.logical_or(predictions_1, predictions_2)

                # plot and save
                

                # get true labels as numpy array
                true_labels = np.zeros(int(duration))
                for _, row in df.iterrows():
                    if row['eventType'] != 'bckg':
                        start = int(row['onset'])
                        end = int((row['onset'] + row['duration']))
                        if start < len(true_labels):
                            true_labels[start:min(end, len(true_labels))] = 1
                


                # plt.plot(predictions)
                # plt.plot(true_labels)

                # plt.savefig(output+subfol+'/'+sess+'/eeg/'+curredf.split('.')[0]+'.png')
                # plt.close()

                # we can begin BIDS conversion process at this point and return into designated directory structure

                # find seizure start times and lengths in the predictions file: every "1" is a seizure and the length of the seizure is the number of consecutive "1"s until the next 0
                # seizure_start = []
                # seizure_length = []
                # i = 0
                # while i < len(predictions):
                #     if predictions[i] == 1:
                #         seizure_start.append(i)
                #         seizure_length.append(0)
                #         while i < len(predictions) and predictions[i] == 1:
                #             seizure_length[-1] += 1
                #             i += 1
                #     else:
                #         i += 1

                # # [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
                # # seizure_start = [4, 9]
                # # seizure_length = [3, 5]
                
                # ret = pd.DataFrame()
                # if not seizure_start:
                #     ret = pd.DataFrame([{
                #     'onset': 0.0,
                #     'duration': float(duration),
                #     'eventType': 'bckg',
                #     'confidence': 1.0,
                #     'channels': 'n/a',
                #     'dateTime': datetime,
                #     'recordingDuration': float(duration)
                # }])
                # else:
                #     ret = pd.DataFrame([{
                #     'onset': float(seizure_start[i]),
                #     'duration': float(seizure_length[i]),
                #     'eventType': 'sz',
                #     'confidence': 1.0,
                #     'channels': 'n/a',
                #     'dateTime': datetime,
                #     'recordingDuration': float(duration)
                # } for i in range(len(seizure_start))])
                
                # ret.to_csv(output +subfol+'/'+sess+'/eeg/'+curredf.split('.')[0]+'.tsv', sep='\t', index=False)



                # % Not sure if evrything below this point is necessary %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


                predictions = predictions[:int(duration)].astype(bool)
                true_labels = true_labels[:int(duration)].astype(bool)
                # predictions = [x == 1 for x in predictions]
                # true_labels = [x == 1 for x in true_labels]
                fs = 1  
                ref_annotation = Annotation(true_labels, fs)
                hyp_annotation = Annotation(predictions, fs)

                params = scoring.EventScoring.Parameters(
                    toleranceStart=30,      # Allow a 30-second early detection margin
                    toleranceEnd=60,        # Allow a 60-second late detection margin
                    minOverlap=0,         # Minimum overlap for a true positive event
                    maxEventDuration=5*60,  # Max duration of an event (5 minutes)
                    minDurationBetweenEvents=90  # Merge events separated by less than 90 seconds
                )

                event_scores = scoring.EventScoring(ref_annotation, hyp_annotation, params)

                # nan handling
                if np.isnan(event_scores.precision):
                    event_scores.precision = 0.
                    event_scores.f1 = 0.

                if np.isnan(event_scores.sensitivity):
                    event_scores.sensitivity = 0.
                    event_scores.f1 = 0.

                if np.isnan(event_scores.f1):
                    event_scores.f1 = 0.

                if np.isnan(event_scores.fpRate):
                    event_scores.fpRate = 0.

                subject_sensitivity.append(event_scores.sensitivity)
                subject_precision.append(event_scores.precision)
                subject_f1.append(event_scores.f1)
                subject_fpRate.append(event_scores.fpRate)


                fig = visualization.plotEventScoring(ref_annotation, hyp_annotation, params)
                fig.savefig(output+subfol+'/'+sess+'/eeg/'+curredf.split('.')[0]+'.png')
        
        
        if subject_sensitivity:
            overall_sensitivity.append(np.mean(subject_sensitivity))
        if subject_precision:
            overall_precision.append(np.mean(subject_precision))
        if subject_f1:
            overall_f1.append(np.mean(subject_f1))
        if subject_fpRate:
            overall_fpRate.append(np.mean(subject_fpRate))

    with open(output+'results.txt', 'w') as f:
        f.write(f"Overall sensitivity: {np.mean(overall_sensitivity)}\n")
        f.write(f"Overall precision: {np.mean(overall_precision)}\n")
        f.write(f"Overall f1: {np.mean(overall_f1)}\n")
        f.write(f"Overall fpRate: {np.mean(overall_fpRate)}\n")


if __name__ == '__main__':
    pipeline(25, 0.65)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('task_id', type=int, help='SGE TASK ID')
    # args = parser.parse_args()

    # match args.task_id:
    #     case 1:
    #         pipeline('tuh-bids', 10, 0.7)
    #         pipeline('siena-bids', 10, 0.7)
    #     case 2:
    #         pipeline('tuh-bids', 20, 0.6)
    #         pipeline('siena-bids', 20, 0.6)
    #     case 3:
    #         pipeline('tuh-bids', 20, 0.7)
    #         pipeline('siena-bids', 20, 0.7)
    #     case 4:
    #         pipeline('tuh-bids', 30, 0.5)
    #         pipeline('siena-bids', 30, 0.5)
    #     case 5:
    #         pipeline('tuh-bids', 30, 0.6)
    #         pipeline('siena-bids', 30, 0.6)
    #     case 6:
    #         pipeline('tuh-bids', 30, 0.7)
    #         pipeline('siena-bids', 30, 0.7)
    #     case 7:
    #         pipeline('tuh-bids', 20, 0.5)
    #         pipeline('siena-bids', 20, 0.5)
    

    # results = szcore_evaluation.evaluate.evaluate_dataset('/projectnb/seizuredet/Sz-challenge/reftest/', '/projectnb/seizuredet/Sz-challenge/hyptest/', '/projectnb/seizuredet/Sz-challenge/output.json')
    # print(results)




    # 1. Training:
    # - Choose 30-40 subjects from TUH and Siena and remaining from the validation splits and a couple from TUH testing set for retraining/validation
    # - Change names of models and save paths -- don't overwrite existing models (save)
    # 2. Dockerfile
    # - Assume getting raw .edf files, do BIDS conversion using epilepsy2bids library, then save as BIDS output file with .tsv in it
    # 3. Meeting Tuesday 10am
    # 4. Email Archana
    # 5. Work on Abstract next week (take abstracts from any papers from the lab, change up how we train on failures [find term])