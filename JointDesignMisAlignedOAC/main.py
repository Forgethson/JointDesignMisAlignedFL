#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import pickle
import random
import time
import copy
from load_data import Load_Data
import cvxpy as cp

from joint import optimize
from torchvision import datasets, transforms

from models.Fed import *
from models.Nets import CNNMnist, CNNCifar2, CNNCifar1
from models.Update import LocalUpdate
from models.test import test_img
from utils.options import args_parser
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def Main(args, method):
    setup_seed(args.seed)
    M = max(int(args.frac * args.M_Prime), 1)  # M = 4
    L = args.L
    f = np.random.randn(5, 1) + np.random.randn(5, 1) * 1j
    f = f / np.linalg.norm(f, 2) * np.sqrt(1)
    f_norm2 = np.linalg.norm(f, 2) ** 2
    args.f_norm2 = f_norm2

    dict_users, dataset_train, dataset_test = Load_Data(args)
    args.testDataset = dataset_test
    K = np.zeros(args.M_Prime)
    for i, item in enumerate(dict_users.keys()):
        K[i] = len(dict_users[item])
    args.K = K

    V = np.zeros((L, M * L))
    v = np.ones((1, M))
    for i in range(L):
        V[i, np.arange(M) + i * M] = v
    P_entry = np.ones(M)
    P = np.zeros((M * (L + 1) - 1, M * L))
    for i in range(M * L):
        P[np.arange(M) + i, i] = P_entry[np.mod(i, M)]
    Q = V.T.conj() @ V

    result = {}

    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar2(args=args).to(args.device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'Fmnist'):
        net_glob = CNNMnist(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')

    w_glob = net_glob.state_dict()  # copy weights
    d = 0
    for item in w_glob.keys():
        d = d + int(np.prod(w_glob[item].shape))
    print(f"\nThe method = {method}")
    print("Running args.EsN0dB =", args.EsN0dB)
    print(f'Total Number of Parameters={d}')
    print(f'The Dataset = {args.dataset}')
    print(f'The Learning Rate = {args.lr}')
    print(f'The local E = {args.local_ep}')
    print(f'The Epoch = {args.epochs}')
    print(f'The iid = {args.iid}')

    net_glob.train()
    print("============================== Federated Learning ... ...")
    # training
    loss_train = []
    acc_store = []
    w_glob = net_glob.state_dict()  # initial global weights
    for iter in range(args.epochs):
        # record the running time of an iteration
        startTime = time.time()
        if iter == 0:
            net_glob.eval()
            acc_test, _ = test_img(net_glob, dataset_test, args)
            acc_store.append(acc_test.numpy())
            net_glob.train()
        history_dict = net_glob.state_dict()

        idxs_users = np.random.choice(range(args.M_Prime), M, replace=False)
        # ----------------------------------------------------------------------- Local Training
        w_locals = []  # store the local "updates" (the difference) of M devices
        loss_locals = []  # store the local training loss
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device), history_dict=history_dict,
                                  user_idx=idx, args=args)
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # ----------------------------------------------------------------------- Federated Averaging
        if method == 'Noiseless':
            current_dict_var = FedAvg(w_locals, args)
        else:
            current_dict_var = FedAvg_ML(w_locals, args, method, Q, P, V, idxs_users)
        # ----------------------------------------------------------------------- Reconstruct the new model at the PS
        for k in current_dict_var.keys():
            w_glob[k] = history_dict[k] + current_dict_var[k]
        # load new model
        net_glob.load_state_dict(w_glob)

        # print training loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print(f'Round {iter:3d}, Average loss {loss_avg:.3f}, Time Cosumed {time.time() - startTime:.3f}')
        loss_train.append(loss_avg)

        # testing
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        acc_store.append(acc_test.numpy())
        net_glob.train()
    result['train_loss'] = np.asarray(loss_train)
    result['test_acc'] = np.asarray(acc_store)
    return result, net_glob


if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    method = args.method
    final_result = Main(args, method)
    print(final_result)
