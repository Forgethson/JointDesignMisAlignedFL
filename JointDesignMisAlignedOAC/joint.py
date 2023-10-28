"""
作者：hp
日期：2022年08月26日
"""
import cvxpy as cp
import numpy as np


def optimize(args, M, N, h, Q, P, dd, Vm2, cnt):
    f_norm2 = args.f_norm2
    P0 = args.P0
    L = args.L
    while np.max(Vm2) > 10:
        Vm2 = Vm2 * 1e-1
    while np.max(Vm2) < 1e-5:
        Vm2 = Vm2 * 1e1
        cnt += 1
    noisePowerVec = 1 / dd
    DzVec = np.tile(noisePowerVec, [1, L + 1])
    Dz = args.f_norm2 * np.diag(DzVec[0][:-1])
    W = P.T.conj() @ np.linalg.inv(Dz) @ P
    cc = np.linalg.inv(np.linalg.cholesky(W))
    W_inv = np.dot(cc.T.conj(), cc)
    A = W_inv * Q.T.conj()
    D_A = np.zeros((M, M))
    for i in range(0, M * L, M):
        for j in range(0, M * L, M):
            D_A += A[i:i + M, j:j + M]
    while np.max(np.linalg.eig(D_A)[0]) > 1e4:
        D_A = D_A * 1e-1

    Vm2 = Vm2[:, np.newaxis]
    P0_Vec = P0 / Vm2

    c_ran = np.random.randn(1, M) + np.random.randn(1, M) * 1j
    d_ran = np.random.randn(1, M) + np.random.randn(1, M) * 1j
    c = np.concatenate((np.real(c_ran), np.imag(c_ran)), axis=0)
    d = np.concatenate((np.real(d_ran), np.imag(d_ran)), axis=0)
    for i in range(M):
        if np.abs(c_ran[0, i]) < 1e-3:
            c_ran[0, i] = c_ran[0, i] / np.abs(c_ran[0, i]) * 1e-3
        if np.abs(d_ran[0, i]) < 1e-3:
            d_ran[0, i] = d_ran[0, i] / np.abs(d_ran[0, i]) * 1e-3

    xx = cp.Variable((M, 1), complex=True)
    ff = cp.Variable((N, 1), complex=True)

    for idx in range(args.SCA_I_max):
        obj = cp.Minimize(cp.quad_form(xx, D_A))
        constraints = [
            np.linalg.norm(c[:, i], 2) ** 2 * np.linalg.norm(d[:, i], 2) ** 2 +
            2 * np.linalg.norm(d[:, i], 2) ** 2 * c[:, i].T @ ([cp.real(xx[i, 0]), cp.imag(xx[i, 0])] - c[:, i]) +
            2 * np.linalg.norm(c[:, i], 2) ** 2 * d[:, i].T @ (
                    [cp.real(cp.conj(ff.T) @ h[:, i]), cp.imag(cp.conj(ff.T) @ h[:, i])] - d[:, i])
            >= cp.inv_pos(P0_Vec[i, 0]) for i in range(M)]

        constraints += [
            np.linalg.norm(c[:, i], 2) ** 2 * np.linalg.norm(d[:, i], 2) ** 2 +
            2 * np.linalg.norm(d[:, i], 2) ** 2 * c[:, i].T @ ([cp.real(xx[i, 0]), cp.imag(xx[i, 0])] - c[:, i]) +
            2 * np.linalg.norm(c[:, i], 2) ** 2 * d[:, i].T @ (
                    [cp.real(cp.conj(ff.T) @ h[:, i]), cp.imag(cp.conj(ff.T) @ h[:, i])] - d[:, i])
            <= cp.max(Vm2) for i in range(M)]

        constraints += [cp.norm(ff, 2) ** 2 <= f_norm2]

        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(verbose=False)
        except cp.SolverError:
            print("Solver 'MOSEK' failed.")
            xx.value = np.random.randn(M, 1) + np.random.randn(M, 1) * 1j
            ff.value = np.random.randn(N, 1) + np.random.randn(N, 1) * 1j
            c_ran = np.random.randn(1, M) + np.random.randn(1, M) * 1j
            d_ran = np.random.randn(1, M) + np.random.randn(1, M) * 1j
            c = np.concatenate((np.real(c_ran), np.imag(c_ran)), axis=0)
            d = np.concatenate((np.real(d_ran), np.imag(d_ran)), axis=0)
            for i in range(M):
                if np.abs(c_ran[0, i]) < 1e-3:
                    c_ran[0, i] = c_ran[0, i] / np.abs(c_ran[0, i]) * 1e-3
                if np.abs(d_ran[0, i]) < 1e-3:
                    d_ran[0, i] = d_ran[0, i] / np.abs(d_ran[0, i]) * 1e-3
            # break

        if prob.status == 'infeasible':
            print('problem became infeasible at iteration{}'.format(idx + 1))
            xx.value = np.random.randn(M, 1) + np.random.randn(M, 1) * 1j
            ff.value = np.random.randn(N, 1) + np.random.randn(N, 1) * 1j
            c_ran = np.random.randn(1, M) + np.random.randn(1, M) * 1j
            d_ran = np.random.randn(1, M) + np.random.randn(1, M) * 1j
            c = np.concatenate((np.real(c_ran), np.imag(c_ran)), axis=0)
            d = np.concatenate((np.real(d_ran), np.imag(d_ran)), axis=0)
            for i in range(M):
                if np.abs(c_ran[0, i]) < 1e-3:
                    c_ran[0, i] = c_ran[0, i] / np.abs(c_ran[0, i]) * 1e-3
                if np.abs(d_ran[0, i]) < 1e-3:
                    d_ran[0, i] = d_ran[0, i] / np.abs(d_ran[0, i]) * 1e-3

        gap = 0
        for i in range(M):
            gap_c = np.linalg.norm(np.array([np.real(xx[i, 0].value), np.imag(xx[i, 0].value)]) - c[:, i], 2)
            gap_d = np.linalg.norm(
                np.array(
                    [float(np.real(ff.value.T.conj() @ h[:, i])), float(np.imag(ff.value.T.conj() @ h[:, i]))]) - d[:,
                                                                                                                  i], 2)
            gap = gap_d + gap_c
            c[:, i] = np.array([np.real(xx[i, 0].value), np.imag(xx[i, 0].value)])
            d[:, i] = [float(np.real(ff.value.T.conj() @ h[:, i])), float(np.imag(ff.value.T.conj() @ h[:, i]))]

        if gap <= args.epsilon:
            break

    result_obj = prob.value
    result_xx = xx.value
    result_ff = ff.value
    pp = ((1 / result_xx).T.conj() / (result_ff.T.conj() @ h)).T.conj()
    pp2 = np.abs(pp) ** 2
    obj_SCA = result_obj / L

    print(f'Avg_obj_SCA = {obj_SCA}')
    a = 1 / min(pp2)
    b = max(Vm2)
    if abs(b - a) <= 1e-7 or a <= b:
        print('True')
        args.count = 0
    else:
        print('False')
        args.count = args.count + 1
        Vm2 = Vm2.reshape(-1)
        if args.count >= 3:
            result_obj, result_ff, pp, cnt = optimize(args, M, N, h, Q, P, dd, Vm2 * 10, cnt + 1)
        else:
            result_obj, result_ff, pp, cnt = optimize(args, M, N, h, Q, P, dd, Vm2, cnt)
    return result_obj, result_ff, pp, cnt
