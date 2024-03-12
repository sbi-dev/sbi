import torch
from torch import Tensor, nn

from sbi.neural_nets.ratio_estimators.base import RatioEstimator


class ClassifierRatio(RatioEstimator):
    r"""classifier- based density ratio estimator."""

    def forward(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Return the logits of the batched (input,condition) pairs.

        Args:
            input: Inputs to evaluate the log probability on of shape
                    (*batch_shape, input_size).
            condition: Conditions of shape (*batch_shape, *condition_shape).

        Raises:
            RuntimeError: If batch_shapes don't match.

        Returns:
            Sample-wise logits.

        Note:
            This function should support PyTorch's automatic broadcasting. This means
            the function should behave as follows for different input and condition
            shapes:
            - (batch_size, input_size) + (*condition_shape) -> (batch_size,)
            - (batch_size, input_size) + (batch_size, *condition_shape) -> (batch_size,)
            - (batch_size1, input_size) + (batch_size2, *condition_shape)
                                                  -> RuntimeError i.e. not broadcastable
        """

        self._check_condition_shape(condition)
        condition_dims = len(self._condition_shape)

        # PyTorch's automatic broadcasting
        batch_shape_in = input.shape[:-1]
        batch_shape_cond = condition.shape[:-condition_dims]
        assert batch_shape_cond == batch_shape_in or len(batch_shape_cond) == 0, (
            "Batch shapes don't match. "
            f"input: {input.shape}, condition: {condition.shape}."
        )

        condition = condition.expand(batch_shape_in + self._condition_shape)
        # Flatten required by nflows, but now both have the same batch shape
        input = input.reshape(-1, input.shape[-1])
        condition = condition.reshape(-1, *self._condition_shape)

        logits = self.net([input, condition])
        return logits

    def loss(self, input: Tensor, condition: Tensor, labels, **kwargs) -> Tensor:
        r"""Return the loss for training the ratio estimator.

        Args:
            input: Inputs to evaluate the loss on of shape (batch_size, input_size).
            condition: Conditions of shape (batch_size, *condition_shape).
            labels: Labels of shape (batch_size,).

        Returns:
            Loss of shape (batch_size,)
        """
        logits = self.forward(input, condition)
        likelihood = torch.sigmoid(logits).squeeze()
        # Binary cross entropy to learn the likelihood
        return nn.BCELoss()(likelihood, labels)
