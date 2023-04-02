#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Distributed under terms of the MIT license.

"""

"""

import argparse
from dataset_utils import DataLoader
from utils import random_planetoid_splits
from GNN_models import *

import torch
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np
from pytorch_lightning import seed_everything


seed_everything(15)


def RunExp(args, dataset, data, Net, percls_trn, val_lb):

    def train(model, optimizer, data, dprate):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]
        nll = F.nll_loss(out, data.y[data.train_mask])
        loss = nll
        loss.backward()

        optimizer.step()
        del out

    def test(model, data):
        model.eval()
        logits, accs, losses, preds = model(data), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(model(data)[mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses

    appnp_net = Net(dataset, args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    permute_masks = random_planetoid_splits
    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb)

    model, data = appnp_net.to(device), data.to(device)

    optimizer = torch.optim.Adam([{
        'params': model.lin1.parameters(),
        'weight_decay': args.weight_decay, 'lr': args.lr
    },
        {
        'params': model.lin2.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
    },
        {
            'params': model.prop1.parameters(),
        'weight_decay': 0.0, 'lr': 0.01
    }
    ],
        lr=args.lr)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    for epoch in range(args.epochs):
        train(model, optimizer, data, args.dprate)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            TEST = appnp_net.prop1.temp.clone()
            Alpha = TEST.detach().cpu().numpy()

            Gamma_0 = Alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break
    return test_acc, best_val_acc, Gamma_0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--train_rate', type=float, default=0.2)
    parser.add_argument('--val_rate', type=float, default=0.1)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--dprate', type=float, default=0.7)
    parser.add_argument('--C', type=int)
    parser.add_argument('--Init', type=str,
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                        default='WS')
    parser.add_argument('--Gamma', default=None)
    parser.add_argument('--ppnp', default='GPR_prop',
                        choices=['PPNP', 'GPR_prop'])
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--output_heads', default=1, type=int)

    parser.add_argument('--dataset', default='chameleon')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--RPMAX', type=int, default=10)
    parser.add_argument('--net', type=str, choices=['GCN', 'GAT', 'APPNP', 'ChebNet', 'JKNet', 'GPRGNN', 'GPRGNNPRE'],
                        default='GPRGNN')

    args = parser.parse_args()

    gnn_name = args.net
    if gnn_name == 'GCN':
        Net = GCN_Net
    elif gnn_name == 'GAT':
        Net = GAT_Net
    elif gnn_name == 'APPNP':
        Net = APPNP_Net
    elif gnn_name == 'ChebNet':
        Net = ChebNet
    elif gnn_name == 'JKNet':
        Net = GCN_JKNet
    elif gnn_name == 'GPRGNN':
        Net = GPRGNN
    elif gnn_name == 'GPRGNNPRE':
        Net = GPRGNN_PRE

    dname = args.dataset
    dataset, data = DataLoader(dname)

    RPMAX = args.RPMAX
    Init = args.Init

    # Gamma_0 = None
    # here we load the GAMMA learnt from the dense result
    # Gamma_FIX = np.load(f'results/gamma_10_Random_{args.dataset}_0.6.npy')
    
    # here we load the GAMMA learnt from the LP task
    Gamma_FIX = np.load(
        f'gamma_10_Random_{args.dataset}_gprgnn_unsupervised.npy')
    # print(Gamma_FIX.shape)

    alpha = args.alpha

    for train_rate in [0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        # train_rate = args.train_rate
        val_rate = args.val_rate
        percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))
        val_lb = int(round(val_rate*len(data.y)))
        TrueLBrate = (percls_trn*dataset.num_classes+val_lb)/len(data.y)
        print('True Label rate: ', TrueLBrate)

        args.C = len(data.y.unique())

        Results0 = []
        GAMMA = []
        test_acc_list = []
        for RP in tqdm(range(RPMAX)):
            # args.Gamma = Gamma_FIX[RP]
            args.Gamma = Gamma_FIX
            test_acc, best_val_acc, Gamma_0 = RunExp(
                args, dataset, data, Net, percls_trn, val_lb)
            print(test_acc)
            Results0.append([test_acc, best_val_acc, Gamma_0])
            GAMMA.append(Gamma_0)
            test_acc_list.append(test_acc)

        test_acc_mean, val_acc_mean, _ = np.mean(Results0, axis=0) * 100
        test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100
        result_root = './results/'
        np.save(
            result_root+f'gamma_{args.K}_{args.Init}_{args.dataset}_{train_rate}_gprgnn.npy', GAMMA)
        np.save(
            result_root+f'acc_{args.K}_{args.Init}_{args.dataset}_{train_rate}_gprgnn.npy', test_acc_list)
        print(f'{gnn_name} on dataset {args.dataset}, in {RPMAX} repeated experiment:')
        print(
            f'test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} \t val acc mean = {val_acc_mean:.4f}')
