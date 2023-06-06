import numpy as np
import pdb
import copy
from joint import optimize


def per_pkt_transmission_Proposed(args, L, TransmittedSymbols, Q, P, V, idxs_users):
    M = max(int(args.frac * args.M_Prime), 1)
    taus = np.sort(np.random.uniform(0, args.maxDelay, (1, M - 1)))[0]
    taus[-1] = args.maxDelay
    dd = np.zeros(M)
    for idx in np.arange(M):
        if idx == 0:
            dd[idx] = taus[0]
        elif idx == M - 1:
            dd[idx] = 1 - taus[-1]
        else:
            dd[idx] = taus[idx] - taus[idx - 1]

    h = 1 / np.sqrt(2) * (np.random.randn(args.N, M) + np.random.randn(args.N, M) * 1j)
    Symbols_var2 = np.sum(np.power(np.abs(TransmittedSymbols), 2), axis=1) / L
    N = args.N
    obj, ff, pp, cnt = optimize(args, M, N, h, Q, P, dd, Symbols_var2)
    pp = pp * np.sqrt(np.power(10, cnt))

    pp_ex = np.tile(pp, (1, len(TransmittedSymbols[0])))
    TransmittedSymbols_pp = TransmittedSymbols * pp_ex
    SigPower = np.var(TransmittedSymbols_pp, axis=1)
    SigPower_mean = np.mean(SigPower)

    D_entry = h.T.conj() @ ff * pp
    for idx in range(M):
        TransmittedSymbols[idx, :] = TransmittedSymbols[idx, :] * D_entry[idx][0]

    EsN0 = np.power(10, args.EsN0dB / 10.0)
    N0 = SigPower_mean / EsN0
    noisePowerVec = N0 / dd

    # Oversample the received signal
    RepeatedSymbols = np.repeat(TransmittedSymbols, M, axis=1)
    for idx in np.arange(M):
        extended = np.array([np.r_[np.zeros(idx), RepeatedSymbols[idx], np.zeros(M - idx - 1)]])
        if idx == 0:
            samples = extended
        else:
            samples = np.r_[samples, extended]
    samples = np.sum(samples, axis=0)

    # generate noise
    for n in range(N):
        for idx in np.arange(M):
            noise = np.random.normal(loc=0, scale=np.sqrt(N0 / 2 / dd[idx]), size=L + 1) + 1j * np.random.normal(
                loc=0, scale=np.sqrt(N0 / 2 / dd[idx]), size=L + 1)
            if idx == 0:
                AWGNnoise = np.array([noise])
            else:
                AWGNnoise = np.r_[AWGNnoise, np.array([noise])]
        AWGNnoise_reshape = np.reshape(AWGNnoise, (1, M * (L + 1)), 'F')
        if n == 0:
            AWGNnoise_f_reshape = np.array([AWGNnoise_reshape.flatten()])
        else:
            AWGNnoise_f_reshape = np.r_[AWGNnoise_f_reshape, np.array([AWGNnoise_reshape.flatten()])]
    noise_plus_f = ff.conj().T @ AWGNnoise_f_reshape
    samples = samples + noise_plus_f[0][0:-1]

    D = np.ones((M * (L + 1) - 1, M * L), dtype=complex)
    DzVec = np.tile(noisePowerVec, [1, L + 1])
    Dz = np.linalg.norm(ff, 2) ** 2 * np.diag(DzVec[0][:-1])

    MUD = np.matmul(D.conj().T, np.linalg.inv(Dz))
    MUD = np.matmul(MUD, D)
    MUD = np.matmul(np.linalg.inv(MUD), D.conj().T)
    MUD = np.matmul(MUD, np.linalg.inv(Dz))
    MUD = np.matmul(MUD, np.array([samples]).T)
    output = (V @ MUD).reshape(-1)
    output = output * args.lr
    return output / M


def per_pkt_transmission_Ori_ML(args, L, TransmittedSymbols, V, idxs_users):
    M = max(int(args.frac * args.M_Prime), 1)
    taus = np.sort(np.random.uniform(0, args.maxDelay, (1, M - 1)))[0]
    taus[-1] = args.maxDelay
    dd = np.zeros(M)
    for idx in np.arange(M):
        if idx == 0:
            dd[idx] = taus[0]
        elif idx == M - 1:
            dd[idx] = 1 - taus[-1]
        else:
            dd[idx] = taus[idx] - taus[idx - 1]

    h = 1 / np.sqrt(2) * (np.random.randn(args.N, M) + np.random.randn(args.N, M) * 1j)
    TransmittedSymbols_pp = TransmittedSymbols
    SigPower = np.var(TransmittedSymbols_pp, axis=1)
    SigPower_mean = np.mean(SigPower)
    D_entry = h[0, :]
    D_entry = D_entry[:, np.newaxis]
    for idx in range(M):
        TransmittedSymbols[idx, :] = TransmittedSymbols[idx, :] * D_entry[idx][0]
    EsN0 = np.power(10, args.EsN0dB / 10.0)
    N0 = SigPower_mean / EsN0
    noisePowerVec = N0 / dd

    # Oversample the received signal
    RepeatedSymbols = np.repeat(TransmittedSymbols, M, axis=1)
    for idx in np.arange(M):
        extended = np.array([np.r_[np.zeros(idx), RepeatedSymbols[idx], np.zeros(M - idx - 1)]])
        if idx == 0:
            samples = extended
        else:
            samples = np.r_[samples, extended]
    samples = np.sum(samples, axis=0)

    # generate noise
    for idx in np.arange(M):
        noise = np.random.normal(loc=0, scale=np.sqrt(N0 / 2 / dd[idx]), size=L + 1) + 1j * np.random.normal(
            loc=0, scale=np.sqrt(N0 / 2 / dd[idx]), size=L + 1)
        if idx == 0:
            AWGNnoise = np.array([noise])
        else:
            AWGNnoise = np.r_[AWGNnoise, np.array([noise])]

    AWGNnoise_reshape = np.reshape(AWGNnoise, (1, M * (L + 1)), 'F')
    samples = samples + AWGNnoise_reshape[0][0:-1]

    D = np.ones((M * (L + 1) - 1, M * L), dtype=complex)
    DzVec = np.tile(noisePowerVec, [1, L + 1])
    Dz = np.diag(DzVec[0][:-1])

    MUD = np.matmul(D.conj().T, np.linalg.inv(Dz))
    MUD = np.matmul(MUD, D)
    MUD = np.matmul(np.linalg.inv(MUD), D.conj().T)
    MUD = np.matmul(MUD, np.linalg.inv(Dz))
    MUD = np.matmul(MUD, np.array([samples]).T)
    output = (V @ MUD).reshape(-1)
    output = output * args.lr
    return output / M


def per_pkt_transmission_Ori_AS(args, L, TransmittedSymbols, idxs_users):
    M = max(int(args.frac * args.M_Prime), 1)
    taus = np.sort(np.random.uniform(0, args.maxDelay, (1, M - 1)))[0]
    taus[-1] = args.maxDelay
    dd = np.zeros(M)
    for idx in np.arange(M):
        if idx == 0:
            dd[idx] = taus[0]
        elif idx == M - 1:
            dd[idx] = 1 - taus[-1]
        else:
            dd[idx] = taus[idx] - taus[idx - 1]

    TransmittedSymbols_pp = TransmittedSymbols
    D_entry = np.ones(M, dtype=complex)

    SigPower = np.var(TransmittedSymbols_pp, axis=1)
    SigPower_mean = np.mean(SigPower)
    D_entry = D_entry[:, np.newaxis]

    for idx in range(M):
        TransmittedSymbols[idx, :] = TransmittedSymbols[idx, :] * D_entry[idx][0]
    EsN0 = np.power(10, args.EsN0dB / 10.0)
    N0 = SigPower_mean / EsN0

    # Oversample the received signal
    RepeatedSymbols = np.repeat(TransmittedSymbols, M, axis=1)
    for idx in np.arange(M):
        extended = np.array([np.r_[np.zeros(idx), RepeatedSymbols[idx], np.zeros(M - idx - 1)]])
        if idx == 0:
            samples = extended
        else:
            samples = np.r_[samples, extended]
    samples = np.sum(samples, axis=0)

    # generate noise
    for idx in np.arange(M):
        noise = np.random.normal(loc=0, scale=np.sqrt(N0 / 2 / dd[idx]), size=L + 1) + 1j * np.random.normal(
            loc=0, scale=np.sqrt(N0 / 2 / dd[idx]), size=L + 1)
        if idx == 0:
            AWGNnoise = np.array([noise])
        else:
            AWGNnoise = np.r_[AWGNnoise, np.array([noise])]

    AWGNnoise_reshape = np.reshape(AWGNnoise, (1, M * (L + 1)), 'F')
    samples = samples + AWGNnoise_reshape[0][0:-1]

    MthFiltersIndex = (np.arange(L) + 1) * M - 1
    output = samples[MthFiltersIndex]
    output = output * args.lr
    return output / M
