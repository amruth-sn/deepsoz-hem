import random
from torch import tensor
from dataloader import *
from baselines import *
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import json
import os
# from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def retrain_model(modelname, manifest, pt_list, maxiter = 30, threshold = 0.5, valsize = 17, cvfold=10, use_cuda=False):
    device = 'cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu'

    detloss = nn.CrossEntropyLoss(weight = torch.Tensor([0.185, 0.815]).double().to(device)) 
    save_loc = '/projectnb/seizuredet/Sz-challenge/retrain/baselines/' + modelname + '/'

    if not os.path.exists(save_loc):
        os.makedirs(save_loc, exist_ok=True)

    manifest = pd.read_csv(manifest)
   
    # y = []
    train_losses = []
    val_losses = []
    outer_step = valsize # 5
    pt_list = np.load(pt_list)
    pt_list = pt_list.tolist()
    random.shuffle(pt_list)
    pt_list = pt_list[:9]
    pt_list = pt_list + [
        #  siena
        'sub-00',
        'sub-01',
        'sub-03',
        'sub-05',
        'sub-06',
        'sub-07',
        'sub-09',
        'sub-10',
        'sub-11',
        'sub-12',
        'sub-13',
        'sub-14',
        'sub-16',
        'sub-17',
        # tuh
        'sub-064',
        'sub-078',
        'sub-167',
        'sub-232',
        'sub-245',
        'sub-291',
        'sub-314',
    ]

    random.shuffle(pt_list)


    # for outer_fold in range(fold_range):
    outer_fold = cvfold
    if outer_fold > 5:
        outer_fold = outer_fold - 5
    
    outer_writer = SummaryWriter(f'runs/outer_{cvfold}/')

    test_pts = pt_list[outer_fold*outer_step: (outer_fold+1)*outer_step] 
    # test_pts has 5 patients
    fold_pts = [pt for pt in pt_list if pt not in test_pts]
    # fold_pts has the remaining 25 or so patients
    inner_splits = []
    
    outer_save_loc = '/projectnb/seizuredet/Sz-challenge/retrain/baselines/' + modelname + '/' + f'outer_{cvfold}/'

    if not os.path.exists(outer_save_loc):
        os.makedirs(outer_save_loc, exist_ok=True)

    with open(outer_save_loc + 'status.txt', 'a') as f:
        f.write("\n\nNow retraining on full train with optimal hyperparameter for outer fold " + str(outer_fold) + ' and evaluating on test.\n')

    p = []
    l = []
    with open(outer_save_loc + 'status.txt', 'a') as f:
            f.write("Data loading...\n")
    fold_set = pretrainLoader(fold_pts, manifest)
    fold_loader = DataLoader(fold_set, batch_size=1, shuffle=True, num_workers=1)

    test_set = pretrainLoader(test_pts, manifest)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=1)
    with open(outer_save_loc + 'status.txt', 'a') as f:
            f.write("Data loaded!\n")

    checkpoint = torch.load('/projectnb/seizuredet/Sz-challenge/code/oldmodel.pth.tar', map_location=device)
    best_inner_model = txlstm_szpool(device=device)
    best_inner_model.load_state_dict(checkpoint)
    best_inner_model = best_inner_model.double()
    best_inner_model.to(device)

    lr = 1e-05
    optimizer = torch.optim.Adam(best_inner_model.parameters(), lr=lr)    

    best_inner_model.train()
    for epoch in range(1, maxiter + 1):
        epoch_loss = 0.0
        train_len = len(fold_loader)
        with open(outer_save_loc + 'status.txt', 'a') as f:
                f.write(str(epoch) + '\n')

        for batch_idx, data in enumerate(fold_loader):
            # MAIN TRAINING LOOP

            optimizer.zero_grad()
            inputs = data['buffers']
            det_labels = data['sz_labels'].long().to(device)
            inputs = inputs.to(torch.DoubleTensor()).to(device)
            
            k_pred, hc, _, _  = best_inner_model(inputs)
            
            del inputs
            loss = detloss(k_pred.reshape(-1,2), det_labels.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            del det_labels
        

        epoch_val_loss = 0.
        val_len = len(test_loader)
        for batch_idx,data in enumerate(test_loader):

            # EVALUATING FOR LOSS EACH EPOCH

            optimizer.zero_grad()
            with torch.no_grad():
                optimizer.zero_grad()
                inputs = data['buffers']
                det_labels = data['sz_labels'].long().to(device)

                inputs = inputs.to(torch.DoubleTensor()).to(device)
                
                k_pred, psoz, ysoz, _  = best_inner_model(inputs)
                

                loss = detloss(k_pred.reshape(-1,2), det_labels.reshape(-1))
                epoch_val_loss += loss.item()
                del inputs, det_labels
                
        epoch_loss = epoch_loss/train_len
        epoch_val_loss = epoch_val_loss/val_len
        train_losses.append(epoch_loss)
        val_losses.append(epoch_val_loss)
        outer_writer.add_scalar('Final/train_loss', epoch_loss, epoch)
        outer_writer.add_scalar('Final/test_loss', epoch_val_loss, epoch)
        

    
    for batch_idx, data in enumerate(test_loader):
        
        # EVALUATION ON OUTER TESTING SET

        with torch.no_grad():
            inputs = data['buffers']
            det_labels = data['sz_labels'].long().to(device)

            inputs = inputs.to(torch.DoubleTensor()).to(device)
            if modelname == 'cnnblstm':
                k_pred, psoz, ysoz = best_inner_model(inputs)
            elif modelname == 'txlstm':
                k_pred, psoz, ysoz, _ = best_inner_model(inputs)

            pred_labels = k_pred.squeeze()
            pred_labels = torch.softmax(pred_labels, dim=1)
            pred_labels = pred_labels.float()

            p.append(pred_labels.cpu().numpy()) 
            l.append(det_labels.cpu().numpy())
            del inputs, det_labels
    
    p = np.array(p)
    l = np.array(l)
    l = np.squeeze(l)
    l = l.flatten(order='C')
    p = (p[:, :, 1] > 0.5).astype(int)
    p = p.flatten(order='C')
    tn, fp, fn, tp = confusion_matrix(l, p).ravel()
    specificity = tn / (tn + fp)
    metrics = {'f1': f1_score(l, p),'roc_auc': roc_auc_score(l, p), 'accuracy': accuracy_score(l, p), 'precision': precision_score(l, p), 'recall': recall_score(l, p), 'specificity': specificity}
    # test_pts = test_pts.tolist()
    outer_fold = int(outer_fold)
    outer_split_dict = {'outer_train': fold_pts, 'test': test_pts, 'outer_fold': outer_fold, 'metrics': metrics, 'train_losses': train_losses, 'test_losses': val_losses, 'splits': inner_splits}
    
    torch.save(best_inner_model.state_dict(), outer_save_loc + f'new_retrained_model.pth_{cvfold}.tar')
    
    with open(outer_save_loc+f'outer_{cvfold}'+'.json', 'w') as f:
        json.dump(outer_split_dict, f, indent=4)

    # METRICS STORED AND DUMPED
    with open(outer_save_loc + 'status.txt', 'a') as f:
        f.write("\n\nDone with outer fold " + str(outer_fold) + '. Metrics stored and dumped.\n\n################################################################\n\n')

    
    del p, l
    
    # print('Done with model ' + modelname)

    outer_writer.add_scalar('Final/f1_score', metrics['f1'], 0)
    outer_writer.add_scalar('Final/roc_auc', metrics['roc_auc'], 0)
    outer_writer.add_scalar('Final/specificity', metrics['specificity'], 0)
    outer_writer.close()
        
    return None