import torch
from torch import Tensor


def get_parameter_mask(model_mask: Tensor, partition: Tensor):
    """takes the binary model mask and returns an extended mask for the model paramters
        dependent on partition of the parameter space

    Args:
        model_mask (Tensor): (batch, len(partition) )
        partition (Tensor): tensor of ints, the partition on the param space

    Returns:
        parameter_mask: (batch, features)
    """
    batchsize = model_mask.size()[0]
    dim = partition.sum()

    parameter_mask = torch.zeros((batchsize, dim), dtype=bool)

    count = 0
    for i, item in enumerate(partition):
        parameter_mask[:, count : count + item] = model_mask[:, i].repeat(item, 1).T
        count += item

    return parameter_mask
