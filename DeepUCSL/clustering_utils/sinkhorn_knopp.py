import numpy as np
import torch

@torch.no_grad()
def distributed_sinkhorn(S, epsilon=0.05, sinkhorn_iterations=15):
    Q = torch.exp(S / epsilon).t() # Q is K-by-B for consistency with notations from the SwAV paper
    B = Q.shape[1] # / world_size  # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.nansum(Q)
    Q /= (sum_Q + 1e-5)

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.nansum(Q, dim=1, keepdim=True)
        Q /= (sum_of_rows + 1e-5)
        Q /= (K + 1e-5)

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.nansum(Q, dim=0, keepdim=True)
        Q /= (B + 1e-5)

    Q *= B  # the columns must sum to 1 so that Q is an assignment
    return Q.t()

def cpu_sk(S, lambda_=1):
    """ Sinkhorn Knopp optimization on CPU
        * stores activations to RAM
        * does matrix-vector multiplies on CPU
        * slower than GPU
    """
    # 1. aggregate inputs:
    N = S.shape[0]
    K = S.shape[1]
    if K == 1:
        return S

    # 2. solve label assignment via sinkhorn-knopp:
    S_posterior = optimize_S_sk(S, K, N, lambda_)
    return S_posterior


def optimize_S_sk(S, K, N, lambda_):
    S_posterior = np.copy(S).T  # now it is K x N
    r = np.ones((K, 1)) / K
    c = np.ones((N, 1)) / N
    S_posterior **= lambda_  # K x N
    inv_K = 1. / K
    inv_N = 1. / N
    err = 1e6
    _counter = 0
    while err > 1e-1:
        r = inv_K / (1e-5 + S_posterior @ c)   # (KxN)@(N,1) = K x 1
        c_new = inv_N / (1e-5 + r.T @ S_posterior).T  # ((1,K)@(KxN)).t() = N x 1
        if _counter % 10 == 0:
            err = np.nansum(np.abs(c / c_new - 1))
        c = c_new
        _counter += 1

    # inplace calculations.
    S_posterior = S_posterior.T
    S_posterior *= c * N
    S_posterior = S_posterior.T
    S_posterior *= r
    S_posterior = S_posterior.T

    return S_posterior

