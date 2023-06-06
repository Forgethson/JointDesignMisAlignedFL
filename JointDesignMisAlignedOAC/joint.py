"""
作者：hp
日期：2022年08月26日
"""
import cvxpy as cp
import numpy as np


def optimize(M, N, L, noisePowerVec, h, f_norm2, V, P0):
    h_real = np.real(h)
    h_imag = np.imag(h)
    h_cat1 = np.concatenate((h_real, h_imag), axis=1)
    h_cat2 = np.concatenate((h_imag, -h_real), axis=1)
    h_cat = np.concatenate((h_cat1, h_cat2), axis=0)
    P_entry = np.ones(M)
    P = np.zeros((M * (L + 1) - 1, M * L))
    for i in range(M * L):
        P[np.arange(M) + i, i] = P_entry[np.mod(i, M)]
    DzVec = np.tile(noisePowerVec, [1, L + 1])
    Dz = f_norm2 * np.diag(DzVec[0][:-1])
    W = P.T.conj() @ np.linalg.inv(Dz) @ P
    c = np.linalg.inv(np.linalg.cholesky(W))
    W_inv = np.dot(c.T.conj(), c)
    Q = V.T.conj() @ V
    A = W_inv * Q.T.conj()
    D_A = np.zeros((M, M))
    for i in range(0, M * L, M):
        for j in range(0, M * L, M):
            D_A += A[i:i + M, j:j + M]

    h_cat_2 = np.concatenate((h_cat[:, M:], h_cat[:, 0:M]), axis=1)
    xx = cp.Variable((2 * M, 1))
    ff = cp.Variable((2 * N, 1))
    xx_cat = xx[0:M] + xx[M:2 * M] * 1j
    obj = cp.Minimize(cp.quad_form(xx_cat, D_A))
    constraints = [xx[i, 0] >= cp.inv_pos(ff.T @ h_cat[:, i] * cp.sqrt(P0)) * 0.5 for i in range(2 * M)]
    constraints += [xx[i, 0] >= cp.inv_pos(ff.T @ h_cat_2[:, i] * cp.sqrt(P0)) * 0.5 for i in range(2 * M)]
    constraints += [ff.T @ h_cat[:, i] >= 0 for i in range(2 * M)]
    constraints += [cp.norm(ff, 2) ** 2 <= f_norm2]
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    result_xx = xx.value
    result_ff = ff.value
    result_ff = result_ff[0:N] + result_ff[N:2 * N] * 1j
    result_xx = result_xx[0:M] + result_xx[M:2 * M] * 1j
    pp = ((1 / result_xx).T.conj() / (result_ff.T.conj() @ h)).T.conj()
    return obj, result_ff, pp