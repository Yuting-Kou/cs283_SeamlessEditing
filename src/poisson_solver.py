import numpy as np

def sor_solver(A, b, omega, x0, eps):
    x = x0
    err = np.linalg.norm(A@x-b)
    cnt = 0
    while (err >= eps):
        for i in range(A.shape[0]):
            tmp = np.dot(A[i], x)
            x[i] += omega * (b[i]-tmp)/A[i][i]
        err = np.linalg.norm(A@x-b)
        cnt += 1
    print(cnt)
    return x

A = np.array([[4,-1,-6,0], [-5, -4, 10, 8], [0, 9, 4, -2], [1, 0, -7, 5]])
b = np.array([2,21,-12,-6])
omega = 0.5
eps = 1e-6

x = sor_solver(A, b, 1, np.zeros(4), eps)

