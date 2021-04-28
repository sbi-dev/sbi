import numpy as np
import torch


def jacobian_in_batch(y, x):
    """ Computes the Jacobian matrix for batched data """
    batch = y.shape[0]
    single_y_size = np.prod(y.shape[1:])
    y = y.view(batch, -1)
    vector = torch.ones(batch).to(y)

    jac = [
        torch.autograd.grad(
            y[:, i], x, grad_outputs=vector, retain_graph=True, create_graph=True
        )[0].view(batch, -1)
        for i in range(single_y_size)
    ]
    jac = torch.stack(jac, dim=1)

    return jac.detach()


def diagonal_hessian(y, xs):
    """ Determines the diagonal Hessian matrix, this isn't much faster then computing
    the full hessian but more memory efficient."""
    assert y.numel() == 1
    if torch.is_tensor(xs):
        xs = [xs]
    diagonal_hessian = []
    # Compute hessian
    for x_i in xs:
        dx_i = torch.autograd.grad(y, x_i, create_graph=True)[0].flatten()
        ddx_i = torch.zeros(dx_i.numel())
        for i in range(x_i.numel()):
            ddx_ii = torch.autograd.grad(dx_i[i], x_i, retain_graph=True)[0].flatten()
            ddx_i[i] = ddx_ii[i].clone().detach()
            del ddx_ii
        diagonal_hessian += [ddx_i.reshape(x_i.shape)]
        del dx_i
        del ddx_i
    return diagonal_hessian


def block_diagonal_hessian(y, xs):
    """ Determines the block diagonal hessian."""
    assert y.numel() == 1
    if torch.is_tensor(xs):
        xs = [xs]
    block_diagonal_hessian = []
    # Compute hessian
    for x_i in xs:
        dx_i = torch.autograd.grad(y, x_i, create_graph=True)[0].flatten()
        ddx_i = torch.zeros((dx_i.numel(), dx_i.numel()))
        for i in range(x_i.numel()):
            ddx_ii = torch.autograd.grad(dx_i[i], x_i, retain_graph=True)[0].flatten()
            ddx_i[i, :] = ddx_ii.clone().detach()
            del ddx_ii
        block_diagonal_hessian += [ddx_i]
        del dx_i
        del ddx_i
    return block_diagonal_hessian


def hessian(y, xs):
    """ Computes the full hessian matrix """
    assert y.numel() == 1
    if torch.is_tensor(xs):
        xs = [xs]
    numel = sum([x.numel() for x in xs])
    hessian = torch.zeros((numel, numel))
    # Compute hessian
    row_idx = 0
    for i, x_i in enumerate(xs):
        dx_i = torch.autograd.grad(y, x_i, create_graph=True)[0].flatten()
        for j in range(x_i.numel()):
            ddx_ij = torch.autograd.grad(dx_i[j], xs[i:], retain_graph=True)
            ddx_ij = torch.cat([x.flatten() for x in ddx_ij])[j:]
            hessian[row_idx, row_idx:] += ddx_ij
            if row_idx + 1 < numel:
                hessian[row_idx + 1 :, row_idx] += ddx_ij[1:]
            del ddx_ij
            row_idx += 1
        del dx_i
    return hessian
