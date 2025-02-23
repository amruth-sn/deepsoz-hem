import random
from torch import tensor
from dataloader import *
from baselines import *
import numpy as np
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

def nested_cv_pretrain(data_root, modelname, manifest, pt_list, maxiter = 30, lr = 5e-05, threshold = 0.5, valsize = 17, cvfold=10, use_cuda=False):

    device = 'cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu'

    detloss = nn.CrossEntropyLoss(weight = torch.Tensor([0.185, 0.815]).double().to(device)) 
    # changed to reflect class weighting
    

    # save_loc = cv_root+'fold'+str(cvfold)+'/baselines/'
    save_loc = '/projectnb/seizuredet/Sz-challenge/repeated_crossval/baselines/' + modelname + '/'

    if not os.path.exists(save_loc):
        os.makedirs(save_loc, exist_ok=True)

    manifest = pd.read_csv(manifest)
   
    # y = []
    train_losses = []
    val_losses = []
    outer_step = valsize # 17
    pt_list = np.load(pt_list)
    # for outer_fold in range(fold_range):
    outer_fold = cvfold
    outer_writer = SummaryWriter(f'runs/outer_{cvfold}/')

    test_pts = pt_list[outer_fold*outer_step: (outer_fold+1)*outer_step] 
    # test_pts has 17 patients
    fold_pts = [pt for pt in pt_list if pt not in test_pts] # the rest of them
    # fold_pts has the remaining 155 something patients
    inner_splits = []
    
    outer_save_loc = '/projectnb/seizuredet/Sz-challenge/repeated_crossval/baselines/' + modelname + '/' + f'outer_{outer_fold}/'

    if not os.path.exists(outer_save_loc):
        os.makedirs(outer_save_loc, exist_ok=True)

    with open(outer_save_loc + 'status.txt', 'a') as f:
            f.write("\nStarting on outer fold " + str(outer_fold))

    # subfold_range = 10
    # for fold in range(subfold_range):
    #     inner_writer = SummaryWriter(f'runs/outer_{outer_fold}/inner_{fold}')
    #     inner_save_loc = '/projectnb/seizuredet/Sz-challenge/repeated_crossval/baselines/' + modelname + '/' + f'outer_{outer_fold}/' + f'inner_{fold}/'
    #     if not os.path.exists(inner_save_loc):
    #         os.makedirs(inner_save_loc, exist_ok=True)
        
    #     inner_step = 15
    #     val_pts = fold_pts[fold*inner_step: (fold+1)*inner_step]
    #     train_pts = [pt for pt in fold_pts if pt not in val_pts]

    #     with open(outer_save_loc + 'status.txt', 'a') as f:
    #         f.write("\nStarting on inner fold " + str(fold) + ' for outer fold ' + str(outer_fold) + '\n')
    #     print(train_pts)
    #     train_set =  pretrainLoader(train_pts, manifest)
    #     val_set =  pretrainLoader(val_pts, manifest)
    #     with open(outer_save_loc + 'status.txt', 'a') as f:
    #         f.write("Data loading...\n")
    #     train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1)
    #     validation_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=1)         
    #     with open(outer_save_loc + 'status.txt', 'a') as f:
    #         f.write("Data loaded!\n")


    #     if modelname == 'cnnblstm':
    #             model = CNN_BLSTM()
    #     elif modelname == 'txlstm':
    #             model = txlstm_szpool(device=device)
    #     else:
    #         model = transformer_lstm(transformer_dropout=0.15, device=device)
        
    #     savename = modelname+'_nested_cv_'+str(lr) + '_'

    #     model.double()
    #     model.to(device)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #     for epoch in range(1, maxiter + 1):
    #         epoch_loss = 0.0
    #         epoch_val_loss = 0.

    #         train_len = len(train_loader)
    #         with open(outer_save_loc + 'status.txt', 'a') as f:
    #             f.write(str(epoch) + '\n')
    #         for batch_idx, data in enumerate(train_loader):
    #             # MAIN TRAINING LOOP

    #             optimizer.zero_grad()
    #             inputs = data['buffers']
    #             det_labels = data['sz_labels'].long().to(device)
    #             inputs = inputs.to(torch.DoubleTensor()).to(device)

    #             if modelname == 'cnnblstm':
    #                 k_pred, hc, _  = model(inputs)
    #             elif modelname == 'txlstm':
    #                 k_pred, hc, _, _  = model(inputs)
    #             else:
    #                 k_pred, hc, _  = model(inputs)


    #             del inputs
    #             loss = detloss(k_pred.reshape(-1,2), det_labels.reshape(-1))

    #             loss.backward()
    #             optimizer.step()

    #             epoch_loss += loss.item()
    #             del det_labels
    #             # if batch_idx%50==0:
    #             #     print('Epoch: ', epoch, ' batch id: ', batch_idx, 'Loss: ', loss.item())

    #         val_len = len(validation_loader)
    #         # val_predictions = []
    #         # val_labels = []
    #         for batch_idx,data in enumerate(validation_loader):

    #             # EVALUATING FOR LOSS EACH EPOCH

    #             optimizer.zero_grad()
    #             with torch.no_grad():
    #                 optimizer.zero_grad()
    #                 inputs = data['buffers']
    #                 det_labels = data['sz_labels'].long().to(device)

    #                 inputs = inputs.to(torch.DoubleTensor()).to(device)
    #                 if modelname == 'cnnblstm':
    #                     k_pred, psoz, ysoz  = model(inputs)
    #                 elif modelname == 'txlstm':
    #                     k_pred, psoz, ysoz, _  = model(inputs)
    #                 else:
    #                     k_pred, psoz, ysoz  = model(inputs)

    #                 loss = detloss(k_pred.reshape(-1,2), det_labels.reshape(-1))
    #                 epoch_val_loss += loss.item()
    #                 del inputs, det_labels
                    
    #         epoch_loss = epoch_loss/train_len
    #         epoch_val_loss = epoch_val_loss/val_len
    #         train_losses.append(epoch_loss)
    #         val_losses.append(epoch_val_loss)

    #         inner_writer.add_scalar('Loss/train_epoch', epoch_loss, epoch)
    #         inner_writer.add_scalar('Loss/val_epoch', epoch_val_loss, epoch)
            
            


    #     # DONE WITH TRAINING ACROSS EPOCHS

    #     val_predictions = []
    #     val_labels = []
    #     for batch_idx,data in enumerate(validation_loader):
    #         # EVALUATION ON VAL SET FOR THRESHOLDING --> No metrics calculated here, potentially change if asked

    #         optimizer.zero_grad()
    #         with torch.no_grad():
    #             optimizer.zero_grad()
    #             inputs = data['buffers']
    #             det_labels = data['sz_labels'].long().to(device)
    #             Nsz = inputs.shape[1]

    #             inputs = inputs.to(torch.DoubleTensor()).to(device)
    #             if modelname == 'cnnblstm':
    #                 k_pred, psoz, ysoz  = model(inputs)
    #             elif modelname == 'txlstm':
    #                 k_pred, psoz, ysoz, _  = model(inputs)
    #             else:
    #                 k_pred, psoz, ysoz  = model(inputs)

    #             pred_labels = k_pred.squeeze()
    #             pred_labels = torch.softmax(pred_labels, dim=1)
    #             pred_labels = pred_labels.float()

    #             val_predictions.append(pred_labels.cpu().numpy()) 
    #             val_labels.append(det_labels.cpu().numpy())

    #             del inputs, det_labels

                
            

    #     best_threshold = 0.5
    #     # best_threshold, best_fpr = 0.5, None
        
    #     val_predictions = np.array(val_predictions)

    #     val_labels = np.array(val_labels)
    #     val_labels = np.squeeze(val_labels)
    #     val_labels = val_labels.flatten(order='C')

    #     preds = (val_predictions[:, :, 1] > 0.5).astype(int)
    #     flatpreds = preds.flatten(order='C')
    #     num_negatives = np.sum(val_labels == 0)
    #     best_fpr = np.sum((flatpreds == 1) & (val_labels == 0)) / num_negatives
    #     # best_f1 = 0

    #     for threshold in np.arange(0.2, 0.8, 0.025):
    #         preds = (val_predictions[:, :, 1] > threshold).astype(int)
    #         flatpreds = preds.flatten(order='C')
    #         fpr = np.sum((flatpreds == 1) & (val_labels == 0)) / num_negatives
    #         # f1 = f1_score(val_labels, flatpreds)
    #         baseline = 120 / 3600.

    #         if fpr < baseline:
    #             # best_f1 = f1
    #             best_fpr = fpr
    #             best_threshold = threshold
    #             break

    #     best_fpr = float(best_fpr)
    #     best_threshold = float(best_threshold)
    #     metricdict = {'inner_train': train_pts, 'val': val_pts, 'inner_fold': fold, 'threshold': best_threshold, 'train_losses': train_losses, 'val_losses': val_losses}
    #     inner_splits.append(metricdict)
        
    #     with open(inner_save_loc+f'inner_{fold}'+'.json', 'w') as f:
    #         json.dump(metricdict, f, indent=4)

    #     with open(outer_save_loc + 'status.txt', 'a') as f:
    #         f.write("Done training and evaluating fold " + str(fold) + ' for outer fold ' + str(outer_fold) + '. Found optimal threshold. \n\n##########################################################\n\n')

    #     train_losses = []
    #     val_losses = []

    #     inner_writer.add_scalar('Metrics/final_threshold', best_threshold, 0)
    #     inner_writer.add_scalar('Metrics/best_fpr', best_fpr, 0)
    #     inner_writer.close()

    #     torch.save(model.state_dict(), inner_save_loc + f'inner_{fold}_best_trained.pth.tar')
    #     del model, optimizer, train_loader, validation_loader


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

    if modelname == 'cnnblstm':
        best_inner_model = CNN_BLSTM()
    elif modelname == 'txlstm':
        best_inner_model = txlstm_szpool(device=device)
    else:
        best_inner_model = transformer_lstm()
    

    best_inner_model.double()
    best_inner_model.to(device)
    optimizer = torch.optim.Adam(best_inner_model.parameters(), lr=lr)

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
            if modelname == 'cnnblstm':
                k_pred, hc, _  = best_inner_model(inputs)
            elif modelname == 'txlstm':
                k_pred, hc, _, _  = best_inner_model(inputs)
            else:
                k_pred, hc, _  = best_inner_model(inputs)
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
                if modelname == 'cnnblstm':
                    k_pred, psoz, ysoz  = best_inner_model(inputs)
                elif modelname == 'txlstm':
                    k_pred, psoz, ysoz, _  = best_inner_model(inputs)
                else:
                    k_pred, psoz, ysoz  = best_inner_model(inputs)

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
    test_pts = test_pts.tolist()
    outer_fold = int(outer_fold)
    outer_split_dict = {'outer_train': fold_pts, 'test': test_pts, 'outer_fold': outer_fold, 'metrics': metrics, 'train_losses': train_losses, 'test_losses': val_losses, 'splits': inner_splits}
    
    torch.save(best_inner_model.state_dict(), outer_save_loc + 'outer_0_best_trained.pth.tar')
    
    with open(outer_save_loc+f'outer_{outer_fold}'+'.json', 'w') as f:
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