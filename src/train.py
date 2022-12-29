from __future__ import absolute_import
from __future__ import print_function

from model import MSCL

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
from load_data import Dataset

import numpy as np
import argparse
import os
import imp
import re
import pickle
import datetime
import random
import math
import copy
from tqdm import tqdm
from sklearn import metrics

def print_metrics_binary(y_true, predictions, verbose=1):
    predictions = np.array(predictions)
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))
    if verbose:
        print("confusion matrix:")
        print(cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
    auroc = metrics.roc_auc_score(y_true, predictions[:, 1])

    (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, predictions[:, 1])
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    if verbose:
        print("accuracy = {}".format(acc))
        print("recall class 0 = {}".format(rec0))
        print("recall class 1 = {}".format(rec1))
        print("AUC of ROC = {}".format(auroc))
        print("AUC of PRC = {}".format(auprc))
        print("min(+P, Se) = {}".format(minpse))

    return {"acc": acc,
            "rec0": rec0,
            "rec1": rec1,
            "auroc": auroc,
            "auprc": auprc,
            "minpse": minpse
           }

def Bootstrap(y_true,y_pred,test_ret):
    N = len(y_true)
    N_idx = np.arange(N)
    K = 1000
    
    auroc = []
    auprc = []
    minpse = []
    acc = []
    for i in range(K):
        boot_idx = np.random.choice(N_idx, N, replace=True)
        boot_true = np.array(y_true)[boot_idx]
        boot_pred = y_pred[boot_idx, :]
        test_ret = print_metrics_binary(boot_true, boot_pred, verbose=0)
        auroc.append(test_ret['auroc'])
        acc.append(test_ret['acc'])

    print('acc %.4f(%.4f)'%(np.mean(acc), np.std(acc)))
    print('auc %.4f(%.4f)'%(np.mean(auroc), np.std(auroc)))
    
def get_loss(y_pred, y_true):
    loss = torch.nn.BCELoss()
    return loss(y_pred, y_true)

device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED) #numpy
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED) # cpu
torch.cuda.manual_seed(RANDOM_SEED) #gpu
torch.backends.cudnn.deterministic=True # cudnn

train_loader = pickle.load(open("./data/dataset.pkl", "rb"))['train_raw']
valid_loader = pickle.load(open("./data/dataset.pkl", "rb"))['val_raw']
test_loader = pickle.load(open("./data/dataset.pkl", "rb"))['test_raw']
path2vec = pickle.load(open("./data/path2vec.pkl", "rb"))

def train(args):
    max_iters = args.max_epoch
    file_name = './data/ml_pipline_v0.pth'
    model = MSCL(input_dim = args.K2, hidden_dim = args.d, d_model = args.d, MHD_num_head = args.NH, d_ff = 256, output_dim = 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    max_roc = 0
    max_prc = 0
    train_loss = []
    train_model_loss = []
    train_decov_loss = []
    valid_loss = []
    valid_model_loss = []
    valid_decov_loss = []
    history = []
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)

    for each_epoch in range(max_iters):
        batch_loss = []
        model_batch_loss = []
        decov_batch_loss = []

        model.train()

        for step, (batch_x, batch_y, batch_name) in enumerate(train_loader):   
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            paths_emb = []
            for name in batch_name:
                path_emb = path2vec[name + ':' + str(0)].reshape(1,args.d)
                for index in range(args.K2-1): 
                    path_emb = np.concatenate((path_emb,path2vec[name + ':' + str(index + 1)].reshape(1,args.d)),axis=0) 
                path_emb = torch.tensor(path_emb, dtype=torch.float32)
                paths_emb.append(path_emb)

            paths_emb = torch.stack(paths_emb).to(device)
            output, decov_loss = model(batch_x, paths_emb, each_epoch, step)

            model_loss = get_loss(output, batch_y.unsqueeze(-1))
            loss = model_loss + 10* decov_loss

            batch_loss.append(loss.cpu().detach().numpy())
            model_batch_loss.append(model_loss.cpu().detach().numpy())
            decov_batch_loss.append(decov_loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print('Epoch %d Batch %d: Train Loss = %.4f'%(each_epoch, step, np.mean(np.array(batch_loss))))
                print('Model Loss = %.4f, Decov Loss = %.4f'%(np.mean(np.array(model_batch_loss)), np.mean(np.array(decov_batch_loss))))
        train_loss.append(np.mean(np.array(batch_loss)))
        train_model_loss.append(np.mean(np.array(model_batch_loss)))
        train_decov_loss.append(np.mean(np.array(decov_batch_loss)))

        batch_loss = []
        model_batch_loss = []
        decov_batch_loss = []

        y_true = []
        y_pred = []
        with torch.no_grad():
            model.eval()
            for step, (batch_x, batch_y, batch_name) in enumerate(valid_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)

                paths_emb = []
                for name in batch_name:
                    path_emb = path2vec[name + ':' + str(0)].reshape(1,args.d)
                    for index in range(args.K2 - 1): 
                        path_emb = np.concatenate((path_emb,path2vec[name + ':' + str(index + 1)].reshape(1,args.d)),axis=0) 
                    path_emb = torch.tensor(path_emb, dtype=torch.float32)
                    paths_emb.append(path_emb)

                paths_emb = torch.stack(paths_emb).to(device)
                output,decov_loss = model(batch_x, paths_emb, each_epoch, step)

                model_loss = get_loss(output, batch_y.unsqueeze(-1))

                loss = model_loss + 1* decov_loss
                batch_loss.append(loss.cpu().detach().numpy())
                model_batch_loss.append(model_loss.cpu().detach().numpy())
                decov_batch_loss.append(decov_loss.cpu().detach().numpy())
                y_pred += list(output.cpu().detach().numpy().flatten())
                y_true += list(batch_y.cpu().numpy().flatten())

        valid_loss.append(np.mean(np.array(batch_loss)))
        valid_model_loss.append(np.mean(np.array(model_batch_loss)))
        valid_decov_loss.append(np.mean(np.array(decov_batch_loss)))

        print("\n==>Predicting on validation")
        print('Valid Loss = %.4f'%(valid_loss[-1]))
        print('valid_model Loss = %.4f'%(valid_model_loss[-1]))
        print('valid_decov Loss = %.4f'%(valid_decov_loss[-1]))
        y_pred = np.array(y_pred)
        y_pred = np.stack([1 - y_pred, y_pred], axis=1)
        ret = print_metrics_binary(y_true, y_pred)
        history.append(ret)
        print()

        #-------------------- test -----------------------
        batch_loss = []
        y_true = []
        y_pred = []
        with torch.no_grad():
            model.eval()
            for step, (batch_x, batch_y, batch_name) in enumerate(test_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)

                paths_emb = []
                for name in batch_name:
                    path_emb = path2vec[name + ':' + str(0)].reshape(1,args.d)
                    for index in range(args.K2 -1): 
                        path_emb = np.concatenate((path_emb,path2vec[name + ':' + str(index + 1)].reshape(1,args.d)),axis=0) 
                    path_emb = torch.tensor(path_emb, dtype=torch.float32)
                    paths_emb.append(path_emb)

                paths_emb = torch.stack(paths_emb).to(device)
                output = model(batch_x, paths_emb, each_epoch, step)[0]

                loss = get_loss(output, batch_y.unsqueeze(-1))
                batch_loss.append(loss.cpu().detach().numpy())
                y_pred += list(output.cpu().detach().numpy().flatten())
                y_true += list(batch_y.cpu().numpy().flatten())

        print("\n==>Predicting on test")
        print('Test Loss = %.4f'%(np.mean(np.array(batch_loss))))
        y_pred = np.array(y_pred)
        y_pred = np.stack([1 - y_pred, y_pred], axis=1)
        test_res = print_metrics_binary(y_true, y_pred)
        print ('testing experimental report:')
        Bootstrap(y_true, y_pred, test_res)
            
    print('=====DONE=====')

parser = argparse.ArgumentParser()
parser.add_argument('--K2', type=int, default=10, help='paths sample size')
parser.add_argument('--d', type=int, default=64, help='embeddings size')
parser.add_argument('--NH', type=int, default=4, help='head of mutli-head attention networks')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--max_epoch', type=int, default=100, help='the number of epochs')

def main():
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
