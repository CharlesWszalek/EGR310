import numpy as np

# Guassian Elimination
def Gauss (A, b):
    n = len(A)

    for k in range(n):
        for i in np.arange(k+1, n):
            factor = A[i, k] / A[k, k]
            A[i][k:n] = A[i][k:n] - factor * A[k][k:n]
            b[i] = b[i] - factor * b[k]

    x = np.zeros(n)
    for i in np.arange(n-1, 0, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x


A = np.array([[1, 0],
              [0, 1]])

b = np.array([1,
              1])

x = Gauss(A, b)

print(x)