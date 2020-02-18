import torch.nn as nn
import torch


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        return (tensor - self.mean) / self.std


def build_inputs_and_contexts(inputs, context, true_context, normalize):
    """
    Formats inputs and context into the correct shape

    Args:
            inputs: Tensor, input variables.
            context: Tensor or None, conditioning variables. If a Tensor, it must have the same
                number or rows as the inputs. If None, the context is ignored.
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
        ]  # append a dimension

    if context is not None:
        if len(context.shape) > 1 and context.shape[0] > 1 and normalize:
            raise ValueError(
                "Only a single context is allowed for log-prob when normalizing the density. "
                "Please use a for-loop over your inputs and contexts."
            )
        if len(context.shape) == 1:
            context = context[
                None,
            ]  # append a dimension
        context = torch.as_tensor(context)
    else:
        context = true_context[
            None,
        ]
    if context.shape[0] != inputs.shape[0]:
        context = context.repeat(inputs.shape[0], 1)

    if inputs.shape[0] != context.shape[0]:
        raise ValueError(
            "Number of input items must be equal to number of context items."
        )

    return inputs, context
