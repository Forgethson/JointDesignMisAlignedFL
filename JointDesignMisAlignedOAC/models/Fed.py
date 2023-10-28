#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
import torch
from Transmission import per_pkt_transmission_Ori_AS, per_pkt_transmission_Ori_ML, \
    per_pkt_transmission_Proposed, per_pkt_transmission_OAC
from utils.Ignore import ToIgnore, flatten


def FedAvg(w, args, flag=0):
    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        # get the receivd signal
        if (flag == 1) and (k not in ToIgnore):
            continue
        w_avg[k] = w_avg[k] * args.lr
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * args.lr
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


# ================================================ Asynchronous AirComp
def FedAvg_ML(w, args, method, Q, P, V, idxs_users):
    L = args.L
    M = len(w)  # number of devices
    # -------------- Extract the symbols (weight updates) from each devices as a numpy array (complex sequence)
    StoreRecover = np.array([])  # record how to transform the numpy arracy back to dictionary
    for m in np.arange(M):
        # extract symbols from one device (layer by layer)
        wEach = w[m]
        eachWeightNumpy = np.array([])
        for k in wEach.keys():
            # The batch normalization layers should be ignroed as they are not weights
            # (can be transmitted reliably to the PS in practice)
            if k in ToIgnore:
                continue
            temp = wEach[k].cpu().numpy()
            temp, unflatten = flatten(temp)
            if m == 0:
                StoreRecover = np.append(StoreRecover, unflatten)
            eachWeightNumpy = np.append(eachWeightNumpy, temp)

        # stack the symbols from different devices ->-> numpy array of shape M * d_symbol = 4 * 10920
        complexSymbols = eachWeightNumpy[0:int(len(eachWeightNumpy) / 2)] + 1j * \
                         eachWeightNumpy[int(len(eachWeightNumpy) / 2):]
        if m == 0:
            TransmittedSymbols = np.array([complexSymbols])
        else:
            TransmittedSymbols = np.r_[TransmittedSymbols, np.array([complexSymbols])]  # ndarray(4, 10920)，active用户的模参
    d_symbol = len(TransmittedSymbols[0])

    # ---------------------------------------------------------------------------------- pkt by pkt transmission
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        numPkt = 42
        lenPkt = int(d_symbol / numPkt)

    elif args.dataset == 'cifar':
        TransmittedSymbols = np.c_[TransmittedSymbols, np.zeros([M, 7])]
        numPkt = 424
        lenPkt = int((d_symbol + 7) / numPkt)

    if method == 'Proposed_SCA':  # proposed
        results = []
        for idx in range(numPkt):
            # transmissted complex symbols in one pkt
            onePkt = TransmittedSymbols[:, (idx * lenPkt):((idx + 1) * lenPkt)]  # (4, 260)
            symbols_received = per_pkt_transmission_Proposed(args, L, copy.deepcopy(onePkt), Q, P, V, idxs_users, idx)
            results.append(symbols_received)

        for idx in range(len(results)):
            output = results[idx]
            if idx == 0:
                ReceivedComplexPkt = output
            else:
                ReceivedComplexPkt = np.append(ReceivedComplexPkt, output)  # 将数据包恢复成原symbol，个数为：(10920, )

        if args.dataset == 'cifar':
            ReceivedComplexPkt = ReceivedComplexPkt[:-7]
        ReceivedPkt = np.append(np.real(ReceivedComplexPkt[:]), np.imag(ReceivedComplexPkt[:]))
        # the last element (0) must be deleted
        # Reconstruct the dictionary from the numpy array
        # run the federated averaging first (to tackle the batched normalization layer)
        w_avg = FedAvg(w, args, 1)

        startIndex = 0
        # idx = 0
        for idx, k in enumerate(w_avg.keys()):
            # only update the non-batched-normalization-layers in w_avg
            if k not in ToIgnore:
                lenLayer = w_avg[k].numel()
                # get data
                ParamsLayer = ReceivedPkt[startIndex:(startIndex + lenLayer)]
                # reshape
                ParamsLayer_reshaped = StoreRecover[idx](ParamsLayer)
                # convert to torch in cuda()
                w_avg[k] = torch.from_numpy(ParamsLayer_reshaped).cuda()
                startIndex += lenLayer
                # idx += 1

    elif method == 'Ori_ML':  # Ori_ML
        results = []
        for idx in range(numPkt):
            onePkt = TransmittedSymbols[:, (idx * lenPkt):((idx + 1) * lenPkt)]  # (4, 260)
            symbols_received = per_pkt_transmission_Ori_ML(args, L, copy.deepcopy(onePkt), V, idxs_users)
            results.append(symbols_received)
        for idx in range(len(results)):
            output = results[idx]
            if idx == 0:
                ReceivedComplexPkt = output
            else:
                ReceivedComplexPkt = np.append(ReceivedComplexPkt, output)
        if args.dataset == 'cifar':
            ReceivedComplexPkt = ReceivedComplexPkt[:-7]
        ReceivedPkt = np.append(np.real(ReceivedComplexPkt[:]), np.imag(ReceivedComplexPkt[:]))
        w_avg = FedAvg(w, args, 1)
        startIndex = 0
        for idx, k in enumerate(w_avg.keys()):
            if k not in ToIgnore:
                lenLayer = w_avg[k].numel()
                ParamsLayer = ReceivedPkt[startIndex:(startIndex + lenLayer)]
                ParamsLayer_reshaped = StoreRecover[idx](ParamsLayer)
                w_avg[k] = torch.from_numpy(ParamsLayer_reshaped).cuda()
                startIndex += lenLayer

    elif method == 'Ori_AS':  # aligned_sample estimator
        results = []
        for idx in range(numPkt):  # 对于所有用户的第idx个数据包来说
            onePkt = TransmittedSymbols[:, (idx * lenPkt):((idx + 1) * lenPkt)]  # (4, 260)
            symbols_received = per_pkt_transmission_Ori_AS(args, L, copy.deepcopy(onePkt), idxs_users)
            results.append(symbols_received)
        for idx in range(len(results)):
            output = results[idx]
            if idx == 0:
                ReceivedComplexPkt = output
            else:
                ReceivedComplexPkt = np.append(ReceivedComplexPkt, output)
        if args.dataset == 'cifar':
            ReceivedComplexPkt = ReceivedComplexPkt[:-7]
        ReceivedPkt = np.append(np.real(ReceivedComplexPkt[:]), np.imag(ReceivedComplexPkt[:]))
        w_avg = FedAvg(w, args, 1)
        startIndex = 0
        for idx, k in enumerate(w_avg.keys()):
            if k not in ToIgnore:
                lenLayer = w_avg[k].numel()
                ParamsLayer = ReceivedPkt[startIndex:(startIndex + lenLayer)]
                ParamsLayer_reshaped = StoreRecover[idx](ParamsLayer)
                w_avg[k] = torch.from_numpy(ParamsLayer_reshaped).cuda()
                startIndex += lenLayer

    elif method == 'OAC':
        results = []
        for idx in range(numPkt):
            onePkt = TransmittedSymbols[:, (idx * lenPkt):((idx + 1) * lenPkt)]  # (4, 260)
            symbols_received = per_pkt_transmission_OAC(args, L, copy.deepcopy(onePkt), V, idxs_users)
            results.append(symbols_received)
        for idx in range(len(results)):
            output = results[idx]
            if idx == 0:
                ReceivedComplexPkt = output
            else:
                ReceivedComplexPkt = np.append(ReceivedComplexPkt, output)
        if args.dataset == 'cifar':
            ReceivedComplexPkt = ReceivedComplexPkt[:-7]
        ReceivedPkt = np.append(np.real(ReceivedComplexPkt[:]), np.imag(ReceivedComplexPkt[:]))
        w_avg = FedAvg(w, args, 1)
        startIndex = 0
        for idx, k in enumerate(w_avg.keys()):
            if k not in ToIgnore:
                lenLayer = w_avg[k].numel()
                ParamsLayer = ReceivedPkt[startIndex:(startIndex + lenLayer)]
                ParamsLayer_reshaped = StoreRecover[idx](ParamsLayer)
                w_avg[k] = torch.from_numpy(ParamsLayer_reshaped).cuda()
                startIndex += lenLayer

    return w_avg
