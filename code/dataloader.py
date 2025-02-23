import numpy as np
import torch
from torch.utils.data import Dataset

class testLoader(Dataset):                 
    def __init__(self, ptlist, dataset):
        self.ptlist = ptlist
        self.dataset = dataset
    

    def __len__(self):
        return len(self.ptlist)
    
    def __getitem__(self, idx):
        fn = self.ptlist[idx]
        xloc = f'/projectnb/seizuredet/Sz-challenge/{self.dataset}/processed/data/{fn}.npy'
        labelsfn = fn.split('.')[0] + '_labels'
        yloc = f'/projectnb/seizuredet/Sz-challenge/{self.dataset}/processed/labels/{labelsfn}.npy'

        X = np.load(xloc)
        Y = np.load(yloc)

        return {'patient numbers': fn,
                'buffers':torch.Tensor(X), #original
                'sz_labels':torch.Tensor(Y),  #sz
               }
    
class pretrainLoader(Dataset):                 
    def __init__(self, ptlist, manifest):
        self.ptlist = ptlist
        self.manifest = manifest
        self.mnlist = [mnitem for  _, mnitem in manifest.iterrows() if mnitem['subject'] in ptlist]
    

    def __len__(self):
        return len(self.mnlist)
    
    def __getitem__(self, idx):
        mnitem = self.mnlist[idx]
        fn = mnitem['fn']
        dataset = mnitem['dataset']
        subject = mnitem['subject']
        session = mnitem['session']
        xloc = f'/projectnb/seizuredet/Sz-challenge/{dataset}/processed/data/{subject}/{session}/{fn}.npy'
        labelsfn = fn.split('.')[0] + '_labels.npy'
        yloc = f'/projectnb/seizuredet/Sz-challenge/{dataset}/processed/labels/{subject}/{session}/{labelsfn}'

        X = np.load(xloc)
        Y = np.load(yloc)

        return {'patient numbers': fn,
                'buffers':torch.Tensor(X), #original
                'sz_labels':torch.Tensor(Y),  #sz
               }


# normalize on the recording level
# z - normalization
# max - min norm





# we get a joint ptlist, then perform train test split, 
# and load data from dataloader in two iterations --> 
# one for tuh patients and one for siena patients





def main():
    return

if __name__ == "__main__":
    main()

