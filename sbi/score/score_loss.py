import torch


def get_slice_vec(x, slice_type):
    v = torch.randn_like(x)
    if slice_type == "unit":
        v = v / v.norm(dim=-1, keepdim=True)
    elif slice_type == "radermacher":
        v = v.sign()
    elif slice_type == "gaussian":
        pass
    else:
        raise NotImplementedError("Unknown slice type")
    return v


def continuous_denoised_loss(score_net, x, context=None):
    """ Score estimation based on denoising autoencoders """
    eps = score_net.eps
    random_t = torch.rand((x.shape[0], 1)) * (1.0 - eps) + eps
    z = torch.randn_like(x)
    mean, std = score_net.marginal_prob(x, random_t)
    perturbed_x = mean + z * std
    score = score_net(perturbed_x, random_t, context=context)
    loss = torch.mean(torch.sum((score * std + z) ** 2, dim=-1))
    return loss


def continuous_score_matching_loss(score_net, x, context=None):
    """ Score matching by l2 loss minimization """
    eps = score_net.eps
    random_t = torch.rand((x.shape[0], 1)) * (1.0 - eps) + eps
    z = torch.randn_like(x)
    mean, std = score_net.marginal_prob(x, random_t)
    x = mean + z * std
    x.requires_grad = True
    y = score_net(x, random_t, context=context)
    l1 = 0.5 * torch.mean(torch.norm(y, dim=-1) ** 2)
    l2 = 0.0
    for i in range(score_net.input_dim):
        grad = torch.autograd.grad(y[:, i].sum(), x, create_graph=True)[0][:, i]
        l2 += grad.mean()
    return l1 + l2


def continuous_sliced_score_matching_loss(
    score_net, x, n_particles=1, slice_type="unit", context=None
):
    eps = score_net.eps
    random_t = torch.rand((x.shape[0], 1)) * (1.0 - eps) + eps
    z = torch.randn_like(x)
    mean, std = score_net.marginal_prob(x, random_t)
    x = mean + z * std
    x = x.repeat(n_particles, 1)
    random_t = random_t.repeat(n_particles, 1)
    if context is not None:
        context.repeat(n_particles, 1)
    x.requires_grad = True
    v = get_slice_vec(x, slice_type)

    y = score_net(x, random_t, context=context)
    l1 = 0.5 * torch.mean(y * v, dim=-1) ** 2
    grad = torch.autograd.grad(torch.sum(y * v), x, create_graph=True)[0]
    l2 = torch.sum(v * grad, -1)

    l1 = l1.reshape(n_particles, -1).mean(0)
    l2 = l2.reshape(n_particles, -1).mean(0)

    loss = torch.mean(l1 + l2)
    return loss


def get_loss_function(name, net, kwargs):
    if name.lower() == "denoised":
        loss_fn = continuous_denoised_loss
    elif name.lower() == "score_matching":
        loss_fn = continuous_score_matching_loss
    elif name.lower() == "sliced_score_matching":
        loss_fn = continuous_sliced_score_matching_loss

    if kwargs is None:
        kwargs = dict()

    def loss(x, context):
        return loss_fn(net, x=x, context=context, **kwargs)

    return loss
