#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
import torch
from Transmission import per_pkt_transmission_Proposed, per_pkt_transmission_Ori_ML, per_pkt_transmission_Ori_AS
from utils.Ignore import ToIgnore, flatten


# ================================================ perfect channel ->->-> no misalignments, no noise

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

        # stack the symbols from different devices
        complexSymbols = eachWeightNumpy[0:int(len(eachWeightNumpy) / 2)] + 1j * \
                         eachWeightNumpy[int(len(eachWeightNumpy) / 2):]
        if m == 0:
            TransmittedSymbols = np.array([complexSymbols])
        else:
            TransmittedSymbols = np.r_[TransmittedSymbols, np.array([complexSymbols])]

    d_symbol = len(TransmittedSymbols[0])

    numPkt = 42
    lenPkt = int(d_symbol / numPkt)

    if method == 'Proposed_SCA':
        results = []
        for idx in range(numPkt):
            # transmissted complex symbols in one pkt
            onePkt = TransmittedSymbols[:, (idx * lenPkt):((idx + 1) * lenPkt)]
            symbols_received = per_pkt_transmission_Proposed(args, L, copy.deepcopy(onePkt), Q, P, V, idxs_users, idx)
            results.append(symbols_received)

        for idx in range(len(results)):
            output = results[idx]
            if idx == 0:
                ReceivedComplexPkt = output
            else:
                ReceivedComplexPkt = np.append(ReceivedComplexPkt, output)

        ReceivedPkt = np.append(np.real(ReceivedComplexPkt[:]), np.imag(ReceivedComplexPkt[:]))
        w_avg = FedAvg(w, args, 1)

    elif method == 'Ori_ML':
        results = []
        for idx in range(numPkt):
            onePkt = TransmittedSymbols[:, (idx * lenPkt):((idx + 1) * lenPkt)]
            symbols_received = per_pkt_transmission_Ori_ML(args, L, copy.deepcopy(onePkt), V, idxs_users)
            results.append(symbols_received)
        for idx in range(len(results)):
            output = results[idx]
            if idx == 0:
                ReceivedComplexPkt = output
            else:
                ReceivedComplexPkt = np.append(ReceivedComplexPkt, output)

        ReceivedPkt = np.append(np.real(ReceivedComplexPkt[:]), np.imag(ReceivedComplexPkt[:]))
        w_avg = FedAvg(w, args, 1)

    elif method == 'Ori_AS':
        results = []
        for idx in range(numPkt):
            onePkt = TransmittedSymbols[:, (idx * lenPkt):((idx + 1) * lenPkt)]
            symbols_received = per_pkt_transmission_Ori_AS(args, L, copy.deepcopy(onePkt), idxs_users)
            results.append(symbols_received)
        for idx in range(len(results)):
            output = results[idx]
            if idx == 0:
                ReceivedComplexPkt = output
            else:
                ReceivedComplexPkt = np.append(ReceivedComplexPkt, output)

        ReceivedPkt = np.append(np.real(ReceivedComplexPkt[:]), np.imag(ReceivedComplexPkt[:]))
        w_avg = FedAvg(w, args, 1)

    return w_avg
