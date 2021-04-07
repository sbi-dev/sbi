import torch
import numpy as np


def gpdfit(x, sorted=True, eps=1e-8, return_quadrature=False):
    """ Pytorch version of PSIS according to https://github.com/avehtari/PSIS/blob/master/py/psis.py """
    if not sorted:
        x, _ = x.sort()
    N = len(x) 
    PRIOR = 3
    M = 30 + int(np.sqrt(N))

    bs = torch.arange(1,M+1)
    bs = 1- np.sqrt(M/(bs-0.5))
    bs /= PRIOR * x[int(N/4 + 0.5)-1]
    bs += 1/x[-1]
    
    
    ks = -bs
    temp = ks[:,None] * x
    ks = torch.log1p(temp).mean(axis=1)
    L = N*(torch.log(-bs / ks) - ks - 1)

    temp = torch.exp(L - L[:,None])
    w = 1/torch.sum(temp, axis=1)
  
    dii = w >= 10 * eps
    if not torch.all(dii):
        w = w[dii]
        bs = bs[dii]
    w /= w.sum()

    # posterior mean for b
    b = torch.sum(bs * w)
    # Estimate for k
    temp = (-b) * x
    temp = torch.log1p(temp)
    k = torch.mean(temp)
    if return_quadrature:
        temp = -x
        temp = bs[:, None] * temp
        temp = torch.log1p(temp)
        ks = torch.mean(temp, axis=1)
    
    # estimate for sigma
    sigma = -k / b * N / (N - 0)
    # weakly informative prior for k
    a = 10
    k = k * N / (N+a) + a * 0.5 / (N+a)
    if return_quadrature:
        ks *= n / (N+a)
        ks += a * 0.5 / (N+a)

    if return_quadrature:
        return k, sigma, ks, w
    else:
        return k, sigma



def 