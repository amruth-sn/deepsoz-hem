import pandas as pd
from dataloader import *
from retrain import *
from baselines import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('cvfold', help="main cross val fold", type=int)
args = parser.parse_args()
cvfold = args.cvfold - 1


manifest = '/projectnb/seizuredet/Sz-challenge/code/manifest.csv'
ptlist = '/projectnb/seizuredet/Sz-challenge/code/newptlist.npy'


# x = nested_cv_pretrain(data_root= '/projectnb/seizuredet/Sz-challenge/tuh-bids/processed/data/',
#                     modelname = 'txlstm',
#                     manifest=manifest,
#                     pt_list=ptlist,
#                     valsize = 17,  
#                     cvfold=cvfold,
#                     use_cuda=True)

x = retrain_model(modelname = 'txlstm',
                    manifest=manifest,
                    pt_list=ptlist,
                    valsize = 5,  
                    cvfold=cvfold,
                    use_cuda=True)
        
        
