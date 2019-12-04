import numpy as np

def sor_solver(A, b, omega, x0, eps):
    x = x0
    err = np.linalg.norm(A@x-b)
    cnt = 0
    while (err >= eps):
        for i in range(A.shape[0]):
            tmp = 0
            for j in range(A.shape[0]):
                if (j != i):
                    tmp += A[i][j] * x[j]
            x[i] += omega * ((b[i]-tmp)/A[i][i] - x[i])
        err = np.linalg.norm(A@x-b)
        cnt += 1
    print(cnt)
    return x


