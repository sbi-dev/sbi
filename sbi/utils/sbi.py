import torch.nn as nn
import torch

# XXX standardize? zscore?
# XXX want to insert it in Sequential
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        # XXX guard against std \sim 0 (epsilon or raise)
        return (tensor - self.mean) / self.std


def build_inputs_and_contexts(inputs, context, true_context, normalize):
    """
    Formats inputs and context into the correct shape

    Args:
        inputs: Tensor, input variables.
        context: Tensor or None, conditioning variables. If a Tensor, it must have the same number or rows as the inputs. If None, the context is ignored.
        true_context: if context=None, replace it with true_context
        normalize:
            If True, we normalize the output density
            by drawing samples, estimating the acceptance
            ratio, and then scaling the probability with it

    Returns:
        inputs, context as torch.tensors
    """
    inputs = torch.as_tensor(inputs)

    if len(inputs.shape) == 1:
        inputs = inputs[
            None,
        ]  # append batch dimension

    if context is not None:

        if len(context.shape) > 1 and context.shape[0] > 1 and normalize:
            # if multiple observations, with snape avoid expensive leakage
            # correction by rejection sampling
            raise ValueError(
                "Only a single context is allowed for log-prob when normalizing the density. "
                "Please use a for-loop over your inputs and contexts."
            )

        if len(context.shape) == 1:
            context = context[
                None,
            ]  # append batch dimension
        context = torch.as_tensor(context)
    else:
        # if observation x1 is not provided, use the x0 used for training
        context = true_context[
            None,
        ]

    if context.shape[0] != inputs.shape[0]:
        # multiple parameters, single observation:
        # repeat the context to match the parameters
        context = context.repeat(inputs.shape[0], 1)

    if inputs.shape[0] != context.shape[0]:
        # catch all remaining errors after shape-mangling above
        # THIS SHOULD NEVER HAPPEN
        raise ValueError(
            "Number of input items must be equal to number of context items."
        )

    return inputs, context
