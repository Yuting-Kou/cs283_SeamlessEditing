import time

import numpy as np
from numba import jit


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print('func:%r took: %2.4f sec' % (f.__name__, te - ts))
        return result

    return timed


def timeit(f):
    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print('func:%r took: %2.4f sec' % (f.__name__, te-ts))
        return result

    return timed


@timeit
def sor_solver(A, b, omega, x0, eps, max_iter=30):
    x = x0
    err = np.linalg.norm(A @ x - b)
    cnt = 0
    diag = A.diagonal()
    while (err >= eps) and cnt < max_iter:
        for i in range(A.shape[0]):
            # tmp = np.dot(A[i], x)
            tmp = np.array(A.data[i]).dot(x[A.rows[i]])
            x[i] += omega * (b[i] - tmp) / diag[i]
            '''
            if i % 10000 == 0:
                print(cnt, i, np.linalg.norm(A@x-b))
            '''
        print(cnt, err)
        err = np.linalg.norm(A @ x - b)
        cnt += 1
    print(cnt)
    return x


@timeit
@jit(nopython=True, fastmath = True)
def sor_solver_jit(data, rows, diag, b, omega, x0, eps = 2e-3, max_iter = 2000):
    x = x0
    err = np.linalg.norm(my_mul(data, rows, x)-b)
    cnt = 0
    eps *= np.sqrt(b.shape[0] * b.shape[1])
    while (err >= eps) and cnt < max_iter:
        for i in range(len(x0)):
            #tmp = np.dot(A[i], x)
            tmp = data[i].dot(x[rows[i]])
            x[i] += omega * (b[i]-tmp)/diag[i]
        err = np.linalg.norm(my_mul(data, rows, x)-b)
        if cnt % 50 == 0:
            print(cnt, err)
        cnt += 1
    print(cnt)
    return x

@jit(nopython = True, fastmath = True, parallel = True)
def my_mul(data, rows, x):
    n = len(x)
    res = np.zeros(x.shape, dtype = np.float64)
    for i in range(n):
        res[i] = np.dot(data[i], x[rows[i]])
    return res

'''
A = np.array([[4,-1,-6,0], [-5, -4, 10, 8], [0, 9, 4, -2], [1, 0, -7, 5]])
b = np.array([2,21,-12,-6])
omega = 0.5
eps = 1e-6

x = sor_solver(A, b, 1, np.zeros(4), eps)
'''
