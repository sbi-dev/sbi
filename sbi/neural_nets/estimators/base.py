# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn


class ConditionalEstimator(nn.Module, ABC):
    r"""Base class for conditional estimators that estimate properties of
    distributions conditional on an input.

    For example, this can be:
    - Conditional density estimator of the posterior $p(\theta|x)$.
    - Conditional density estimator of the likelihood $p(x|\theta)$.
    - Conditional vector field estimator e.g. $\nabla_\theta \log p(\theta|x)$.

    Subclasses of ConditionalEstimator should implement the ``loss(input, condition)``
    method to be compatible with sbi's training procedures.
    """

    def __init__(self, input_shape: Tuple, condition_shape: Tuple) -> None:
        r"""Construct a conditional estimator given shapes.

        Args:
            input_shape: Event shape of the input at which the density is being
                evaluated (and which is also the event_shape of samples).
            condition_shape: Shape of the condition.
        """
        super().__init__()
        self._input_shape = torch.Size(input_shape)
        self._condition_shape = torch.Size(condition_shape)

    @property
    def input_shape(self) -> torch.Size:
        r"""Return the input shape."""
        return self._input_shape

    @property
    def condition_shape(self) -> torch.Size:
        r"""Return the condition shape."""
        return self._condition_shape

    @abstractmethod
    def loss(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Return the loss for training the estimator.

        Args:
            input: Inputs to evaluate the loss on of shape
                `(batch_dim, *input_event_shape)`.
            condition: Conditions of shape `(batch_dim, *event_shape_condition)`.

        Returns:
            Loss of shape (batch_dim,)
        """
        pass

    def _check_condition_shape(self, condition: Tensor):
        r"""This method checks whether the condition has the correct shape.

        Args:
            condition: Conditions of shape `(batch_dim, *event_shape_condition)`.

        Raises:
            ValueError: If the condition has a dimensionality that does not match
                        the expected input dimensionality.
            ValueError: If the shape of the condition does not match the expected
                        input dimensionality.
        """
        exp_condition_shape = self.condition_shape
        if len(condition.shape) < len(exp_condition_shape):
            raise ValueError(
                "Dimensionality of condition is too small and does not match the "
                f"expected dimensionality {len(exp_condition_shape)}. It should "
                f"be compatible with condition_shape {exp_condition_shape}."
            )
        else:
            condition_shape = condition.shape[-len(self.condition_shape) :]
            if condition_shape != exp_condition_shape:
                raise ValueError(
                    f"Shape of condition {condition_shape} does not match the "
                    f"expected input dimensionality {exp_condition_shape}, as "
                    "provided by condition_shape. Please reshape it accordingly."
                )

    def _check_input_shape(self, input: Tensor):
        r"""This method checks whether the input has the correct shape.

        Args:
            input: Inputs to evaluate the log probability on of shape
                    `(sample_dim_input, batch_dim_input, *event_shape_input)`.

        Raises:
            ValueError: If the input has a dimensionality that does not match
                        the expected input dimensionality.
            ValueError: If the shape of the input does not match the expected
                        input dimensionality.
        """
        input_shape = input.shape
        exp_input_shape = self.input_shape
        if len(input_shape) < len(exp_input_shape):
            raise ValueError(
                "Dimensionality of input is too small and does not match the "
                f"expected dimensionality {len(exp_input_shape)}. It should "
                f"be compatible with the provided input_shape {exp_input_shape}."
            )
        else:
            input_shape = input.shape[-len(self.input_shape) :]
            if input_shape != exp_input_shape:
                raise ValueError(
                    f"Shape of input {input_shape} does not match the "
                    f"expected input dimensionality {exp_input_shape}, as "
                    "provided by input_shape. Please reshape it accordingly."
                )


class MaskedConditionalEstimator(nn.Module, ABC):
    r""" """

    def __init__(self, input_shape: Tuple) -> None:
        r"""Construct a conditional estimator given input shape.

        Args:
            input_shape: Event shape of the input at which the density is being
                evaluated (and which is also the event_shape of samples).
        """
        super().__init__()
        self._input_shape = torch.Size(input_shape)

    @property
    def input_shape(self) -> torch.Size:
        r"""Return the input shape."""
        return self._input_shape

    @abstractmethod
    def loss(
        self, input: Tensor, condition_mask: Tensor, edge_mask: Tensor, **kwargs
    ) -> Tensor:
        r"""Return the loss for training the estimator.

        Args:
            input: Inputs to evaluate the loss on of shape
                `(batch_dim, *input_event_shape)`.

        Returns:
            Loss of shape (batch_dim,)
        """
        pass

    def _check_input_shape(self, input: Tensor):
        r"""This method checks whether the input has the correct shape.

        Args:
            input: Inputs to evaluate the log probability on of shape
                    `(sample_dim_input, batch_dim_input, *event_shape_input)`.

        Raises:
            ValueError: If the input has a dimensionality that does not match
                        the expected input dimensionality.
            ValueError: If the shape of the input does not match the expected
                        input dimensionality.
        """
        input_shape = input.shape
        exp_input_shape = self.input_shape
        if len(input_shape) < len(exp_input_shape):
            raise ValueError(
                "Dimensionality of input is too small and does not match the "
                f"expected dimensionality {len(exp_input_shape)}. It should "
                f"be compatible with the provided input_shape {exp_input_shape}."
            )
        else:
            input_shape = input.shape[-len(self.input_shape) :]
            if input_shape != exp_input_shape:
                raise ValueError(
                    f"Shape of input {input_shape} does not match the "
                    f"expected input dimensionality {exp_input_shape}, as "
                    "provided by input_shape. Please reshape it accordingly."
                )


class ConditionalDensityEstimator(ConditionalEstimator):
    r"""Base class for conditional density estimators.

    The density estimator class is a wrapper around neural networks that allows to
    evaluate the `log_prob`, `sample`, and provide the `loss` of $\theta,x$ pairs. Here
    $\theta$ would be the `input` and $x$ would be the `condition`.

    Note:
        We assume that the input to the density estimator is a tensor of shape
        (sample_dim, batch_dim, *input_shape), where input_shape is the dimensionality
        of the input. The condition is a tensor of shape (batch_size, *condition_shape),
        where condition_shape is the shape of the condition tensor.

    """

    def __init__(
        self, net: nn.Module, input_shape: torch.Size, condition_shape: torch.Size
    ) -> None:
        r"""Base class for density estimators.

        Args:
            net: Neural network or any parameterized model that is used to estimate the
                probability density of the input given a condition.
            input_shape: Event shape of the input at which the density is being
                evaluated (and which is also the event_shape of samples).
            condition_shape: Shape of the condition.
        """
        super().__init__(input_shape, condition_shape)
        self.net = net

    @property
    def embedding_net(self) -> Optional[nn.Module]:
        r"""Return the embedding network if it exists."""
        return None

    @abstractmethod
    def log_prob(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Return the log probabilities of the inputs given a condition or multiple
        i.e. batched conditions.

        Args:
            input: Inputs to evaluate the log probability on of shape
                    `(sample_dim_input, batch_dim_input, *event_shape_input)`.
            condition: Conditions of shape
                `(batch_dim_condition, *event_shape_condition)`.

        Raises:
            RuntimeError: If batch_dim_input and batch_dim_condition do not match.

        Returns:
            Sample-wise log probabilities.
        """

        pass

    @abstractmethod
    def loss(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Return the loss for training the density estimator.

        Args:
            input: Inputs to evaluate the loss on of shape
                `(batch_dim, *input_event_shape)`.
            condition: Conditions of shape `(batch_dim, *event_shape_condition)`.

        Returns:
            Loss of shape (batch_dim,)
        """

        pass

    @abstractmethod
    def sample(self, sample_shape: torch.Size, condition: Tensor, **kwargs) -> Tensor:
        r"""Return samples from the density estimator.

        Args:
            sample_shape: Shape of the samples to return.
            condition: Conditions of shape `(batch_dim, *event_shape_condition)`.

        Returns:
            Samples of shape (*sample_shape, batch_dim, *event_shape_input).
        """

        pass

    def sample_and_log_prob(
        self, sample_shape: torch.Size, condition: Tensor, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        r"""Return samples and their density from the density estimator.

        Args:
            sample_shape: Shape of the samples to return.
            condition: Conditions of shape `(batch_dim, *event_shape_condition)`.

        Returns:
            Samples and associated log probabilities.

        Note:
            For some density estimators, computing log_probs for samples is
            more efficient than computing them separately. This method should
            then be overwritten to provide a more efficient implementation.
        """

        samples = self.sample(sample_shape, condition, **kwargs)
        log_probs = self.log_prob(samples, condition, **kwargs)
        return samples, log_probs


class ConditionalVectorFieldEstimator(ConditionalEstimator, ABC):
    r"""Base class for vector field estimators. That primarily includes
    score-based and flow matching models.

    The vector field estimator class is a wrapper around neural networks that allows to
    evaluate the `vector_field`, and provide the `loss` of $\theta,x_o$ pairs. Here
    $\theta$ would be the `input` and $x_o$ would be the `condition`.

    Note:
        We assume that the input to the density estimator is a tensor of shape
        (sample_dim, batch_dim, *input_shape), where input_shape is the dimensionality
        of the input. The condition is a tensor of shape (batch_dim, *condition_shape),
        where condition_shape is the shape of the condition tensor.
    """

    # When implementing custom estimators,
    # the following properties should be set:

    # Whether the score is defined for this estimator.
    # Required for gradient-based methods.
    # It should be set to True only if score is implemented.
    SCORE_DEFINED: bool = True

    # Whether the SDE functions - score, drift and diffusion -
    # are defined for this estimator.
    # Required for SDE sampling.
    SDE_DEFINED: bool = True

    # Whether the marginals are defined for this estimator.
    # Required for iid methods.
    # It should be set to True only if mean_t_fn and std_fn are implemented.
    MARGINALS_DEFINED: bool = True

    def __init__(
        self,
        net: nn.Module,
        input_shape: torch.Size,
        condition_shape: torch.Size,
        t_min: float = 0.0,
        t_max: float = 1.0,
        embedding_net: Optional[nn.Module] = None,
        mean_base: Union[float, Tensor] = 0.0,
        std_base: Union[float, Tensor] = 1.0,
    ) -> None:
        r"""Base class for vector field estimators.

        Args:
            net: Neural network.
            input_shape: Shape of the input.
            condition_shape: Shape of the condition. If not provided, it will assume a
                            1D input.
            t_min: Minimum time for the vector field estimator.
            t_max: Maximum time for the vector field estimator.
            embedding_net: Embedding network, if provided, it will be used to embed the
                condition.
            mean_base: Mean of the base distribution.
            std_base: Standard deviation of the base distribution.
        """
        super().__init__(input_shape, condition_shape)
        self.net = net

        # We assume that the time range is the same for ODE and SDE.
        self.t_min = t_min
        self.t_max = t_max

        # We assume that the base distribution is a Gaussian distribution
        # and that it is the same for ODE and SDE.
        # We store the mean and std of the base distribution in buffers
        # to transfer them to the device automatically when the model is moved.
        self.register_buffer(
            "_mean_base", torch.empty(1, *self.input_shape).fill_(mean_base)
        )
        self.register_buffer(
            "_std_base", torch.empty(1, *self.input_shape).fill_(std_base)
        )
        self._embedding_net = (
            embedding_net if embedding_net is not None else nn.Identity()
        )

    @property
    def embedding_net(self) -> nn.Module:
        r"""Return the embedding network if it exists."""
        return self._embedding_net

    @abstractmethod
    def forward(
        self, input: Tensor, condition: Tensor, time: Tensor, **kwargs
    ) -> Tensor:
        r"""Forward pass of the score estimator.

        Args:
            input: Input variable :math:`\theta_t`.
            condition: Conditioning variable :math:`x_o`.

        Raises:
            NotImplementedError: This method should be implemented by sub-classes.
        """
        ...

    # -------------------------- BASE DISTRIBUTION METHODS --------------------------

    # We assume that the base distribution is a Gaussian distribution
    # and that it is the same for ODE and SDE.

    @property
    def mean_base(self) -> Tensor:
        r"""Mean of the base distribution (the initial noise at time t=T)."""
        return self._mean_base

    @property
    def std_base(self) -> Tensor:
        r"""Standard deviation of the base distribution
        (the initial noise at time t=T)."""
        return self._std_base

    # -------------------------- ODE METHODS --------------------------

    @abstractmethod
    def ode_fn(self, input: Tensor, condition: Tensor, times: Tensor) -> Tensor:
        r"""ODE flow function :math:`v(\theta_t, t, x_o)` of the vector field estimator.

        The target distribution can be sampled from by solving the following ODE:

        .. math::
            d\theta_t = v(\theta_t, t; x_o) dt

        with initial :math:`\theta_1` sampled from the base distribution.

        Args:
            input: variable whose distribution is estimated.
            condition: Conditioning variable.
            t: Time.

        Raises:
            NotImplementedError: This method should be implemented by sub-classes.
        """
        ...

    # -------------------------- SDE METHODS --------------------------

    def score(self, input: Tensor, condition: Tensor, t: Tensor) -> Tensor:
        r"""Time-dependent score function

        .. math::
            s(t, \theta_t; x_o) = \nabla_{\theta_t} \log p(\theta_t | x_o)

        Args:
            input: Input parameters :math:`\theta_t`.
            condition: Conditioning variable :math:`x_o`.
            t: Time.

        Raises:
            NotImplementedError: Score is not implemented for this estimator.
        """
        raise NotImplementedError("Score is not implemented for this estimator.")

    def mean_t_fn(self, times: Tensor) -> Tensor:
        r"""Linear coefficient mean_t of the perturbation kernel expectation
        :math:`\mu_t(t) = E[\theta_t | \theta_0] = \text{mean_t}(t) \cdot \theta_0`
        specifying the "mean factor" at a given time, which is always multiplied by
        :math:`\theta_0` to get the mean of the noise distribution, i.e.,
        :math:`p(\theta_t | \theta_0) = N(\theta_t;
                \text{mean_t}(t)*\theta_0, \text{std_t}(t)).`

        Args:
            times: SDE time variable in [0,1].

        Raises:
            NotImplementedError: Mean_t is not implemented for this estimator.
        """
        raise NotImplementedError("Mean_t is not implemented for this estimator.")

    def std_fn(self, times: Tensor) -> Tensor:
        r"""Standard deviation function std_t(t) of the perturbation kernel at a given
            time,

        .. math::
            p(\theta_t | \theta_0) = N(\theta_t; \text{mean_t}(t) \cdot
            \theta_0, \text{std_t}(t)^2).

        Args:
            times: SDE time variable in [0,1].

        Raises:
            NotImplementedError: Std_t is not implemented for this estimator.
        """
        raise NotImplementedError("Std_t is not implemented for this estimator.")

    def drift_fn(self, input: Tensor, times: Tensor) -> Tensor:
        r"""Drift function :math:`f(t)` of the vector field estimator.

        The drift function :math:`f(t)` and diffusion function :math:`\g(t)`
        enable SDE sampling:

        .. math::
            d\theta_t = [f(t) - g(t)^2 \nabla_{\theta_t} \log p(\theta_t | x_o)]dt
              + \g(t) dW_t

        where :math:`dW_t` is the Wiener process.


        Args:
            input: input parameters :math:`\theta_t`.
            times: SDE time variable in [0,1].

        Raises:
            NotImplementedError: Drift is not implemented for this estimator.

        """
        raise NotImplementedError("Drift is not implemented for this estimator.")

    def diffusion_fn(self, input: Tensor, times: Tensor) -> Tensor:
        r"""Diffusion function :math:`\g(t)` of the vector field estimator.

        The drift function :math:`f(t)` and diffusion function :math:`\g(t)`
        enable SDE sampling:

        .. math::
            d\theta_t = [f(t) - g(t)^2 \nabla_{\theta_t} \log p(\theta_t | x_o)]dt
              + \g(t) dW_t

        where :math:`dW_t` is the Wiener process.

        Args:
            input: input parameters :math:`\theta_t`.
            times: SDE time variable in [0,1].

        Raises:
            NotImplementedError: Diffusion is not implemented for this estimator.
        """
        raise NotImplementedError("Diffusion is not implemented for this estimator.")


class MaskedConditionalVectorFieldEstimator(MaskedConditionalEstimator, ABC):
    r"""Base class for masked vector field estimators. That primarily includes
    score-based and flow matching models.

    The vector field estimator class is a wrapper around neural networks that allows to
    evaluate the `masked_vector_field`, and provide the `loss` of inputs over masks.

    Note:
        We assume that the input to the density estimator is a tensor of shape
        (sample_dim, batch_dim, *input_shape), where input_shape is the dimensionality
        of the input.
    """

    # When implementing custom estimators,
    # the following properties should be set:

    # Whether the score is defined for this estimator.
    # Required for gradient-based methods.
    # It should be set to True only if score is implemented.
    SCORE_DEFINED: bool = True

    # Whether the SDE functions - score, drift and diffusion -
    # are defined for this estimator.
    # Required for SDE sampling.
    SDE_DEFINED: bool = True

    # Whether the marginals are defined for this estimator.
    # Required for iid methods.
    # It should be set to True only if mean_t_fn and std_fn are implemented.
    MARGINALS_DEFINED: bool = True

    def __init__(
        self,
        net: nn.Module,
        input_shape: torch.Size,
        t_min: float = 0.0,
        t_max: float = 1.0,
        mean_base: float = 0.0,
        std_base: float = 1.0,
    ) -> None:
        r"""Base class for masked vector field estimators.

        Args:
            net: Neural network.
            input_shape: Shape of the input.
            t_min: Minimum time for the vector field estimator.
            t_max: Maximum time for the vector field estimator.
            mean_base: Mean of the base distribution.
            std_base: Standard deviation of the base distribution.
        """
        super().__init__(input_shape)
        self.net = net

        # We assume that the time range is the same for ODE and SDE.
        self.t_min = t_min
        self.t_max = t_max

        # We store the mean and std of the base distribution in buffers
        # to transfer them to the device automatically when the model is moved.
        self.register_buffer(
            "_mean_base", torch.empty(1, *self.input_shape).fill_(mean_base)
        )
        self.register_buffer(
            "_std_base", torch.empty(1, *self.input_shape).fill_(std_base)
        )

    def build_conditional_vector_field_estimator(
        self, fixed_condition_mask: Tensor, fixed_edge_mask: Tensor
    ) -> ConditionalVectorFieldEstimator:
        """Returns a callable that behaves like a ConditionalVectorFieldEstimator
        for a fixed condition_mask and edge_mask.
        """

        return MaskedConditionalVectorFieldEstimatorWrapper(
            self, fixed_condition_mask, fixed_edge_mask
        )

    @abstractmethod
    def forward(
        self,
        input: Tensor,
        time: Tensor,
        condition_mask: Tensor,
        edge_mask: Tensor,
        **kwargs,
    ) -> Tensor:
        r"""Forward pass of the masked conditional vector field estimator.

        Args:
            input: Input variables of shape (batch_dim, *input_shape).
            time: Time variable tensor.
            condition_mask: Mask tensor indicating observed/latent nodes.
            edge_mask: Mask tensor indicating edges in the graph.
            **kwargs: Additional keyword arguments.
        """
        ...

    # -------------------------- BASE DISTRIBUTION METHODS --------------------------

    # We assume that the base distribution is a Gaussian distribution
    # and that it is the same for ODE and SDE.

    @property
    def mean_base(self) -> Tensor:
        r"""Mean of the base distribution (the initial noise at time t=T)."""
        return self._mean_base

    @property
    def std_base(self) -> Tensor:
        r"""Standard deviation of the base distribution
        (the initial noise at time t=T)."""
        return self._std_base

    # -------------------------- ODE METHODS --------------------------

    @abstractmethod
    def ode_fn(
        self,
        input: Tensor,
        times: Tensor,
        condition_mask: Tensor,
        edge_mask: Tensor,
    ) -> Tensor:
        r"""ODE flow function :math:`v(\theta_t, t, x_o)` of the vector field estimator.

        The target distribution can be sampled from by solving the following ODE:

        .. math::
            d\theta_t = v(\theta_t, t; x_o) dt

        with initial :math:`\theta_1` sampled from the base distribution.

        Args:
            input: variable whose distribution is estimated.
            t: Time.

        Raises:
            NotImplementedError: This method should be implemented by sub-classes.
        """
        ...

    # -------------------------- SDE METHODS --------------------------

    def score(
        self,
        input: Tensor,
        t: Tensor,
        condition_mask: Tensor,
        edge_mask: Tensor,
    ) -> Tensor:
        r"""Time-dependent score function

        .. math::
            s(t, \theta_t; x_o) = \nabla_{\theta_t} \log p(\theta_t | x_o)

        Args:
            input: Input parameters :math:`\theta_t`.
            t: Time.

        Raises:
            NotImplementedError: Score is not implemented for this estimator.
        """
        raise NotImplementedError("Score is not implemented for this estimator.")

    def mean_t_fn(self, times: Tensor) -> Tensor:
        r"""Linear coefficient mean_t of the perturbation kernel expectation
        :math:`\mu_t(t) = E[\theta_t | \theta_0] = \text{mean_t}(t) \cdot \theta_0`
        specifying the "mean factor" at a given time, which is always multiplied by
        :math:`\theta_0` to get the mean of the noise distribution, i.e.,
        :math:`p(\theta_t | \theta_0) = N(\theta_t;
                \text{mean_t}(t)*\theta_0, \text{std_t}(t)).`

        Args:
            times: SDE time variable in [0,1].

        Raises:
            NotImplementedError: Mean_t is not implemented for this estimator.
        """
        raise NotImplementedError("Mean_t is not implemented for this estimator.")

    def std_fn(self, times: Tensor) -> Tensor:
        r"""Standard deviation function std_t(t) of the perturbation kernel at a given
            time,

        .. math::
            p(\theta_t | \theta_0) = N(\theta_t; \text{mean_t}(t) \cdot
            \theta_0, \text{std_t}(t)^2).

        Args:
            times: SDE time variable in [0,1].

        Raises:
            NotImplementedError: Std_t is not implemented for this estimator.
        """
        raise NotImplementedError("Std_t is not implemented for this estimator.")

    def drift_fn(self, input: Tensor, times: Tensor) -> Tensor:
        r"""Drift function :math:`f(t)` of the vector field estimator.

        The drift function :math:`f(t)` and diffusion function :math:`\g(t)`
        enable SDE sampling:

        .. math::
            d\theta_t = [f(t) - g(t)^2 \nabla_{\theta_t} \log p(\theta_t | x_o)]dt
              + \g(t) dW_t

        where :math:`dW_t` is the Wiener process.


        Args:
            input: input parameters :math:`\theta_t`.
            times: SDE time variable in [0,1].

        Raises:
            NotImplementedError: Drift is not implemented for this estimator.

        """
        raise NotImplementedError("Drift is not implemented for this estimator.")

    def diffusion_fn(self, input: Tensor, times: Tensor) -> Tensor:
        r"""Diffusion function :math:`\g(t)` of the vector field estimator.

        The drift function :math:`f(t)` and diffusion function :math:`\g(t)`
        enable SDE sampling:

        .. math::
            d\theta_t = [f(t) - g(t)^2 \nabla_{\theta_t} \log p(\theta_t | x_o)]dt
              + \g(t) dW_t

        where :math:`dW_t` is the Wiener process.

        Args:
            input: input parameters :math:`\theta_t`.
            times: SDE time variable in [0,1].

        Raises:
            NotImplementedError: Diffusion is not implemented for this estimator.
        """
        raise NotImplementedError("Diffusion is not implemented for this estimator.")


class MaskedConditionalVectorFieldEstimatorWrapper(ConditionalVectorFieldEstimator):
    r"""Base class for wrapper to adapt masked conditional vector field estimator
    to a conditional vector field estimator interface.

    This wrapper adapts a MaskedConditionalVectorFieldEstimator to the
    ConditionalVectorFieldEstimator interface by taking a fixed edge and condition
    mask. It assembles the input and condition tensors into the full input expected
    by the masked estimator, calls the masked estimator, and then disassembles the
    output to return only the relevant latent part. This allows using a masked estimator
    as if it were a standard conditional estimator for a fixed mask configuration.
    """

    def __init__(
        self,
        original_estimator: MaskedConditionalVectorFieldEstimator,
        fixed_condition_mask: Optional[Tensor] = None,
        fixed_edge_mask: Optional[Tensor] = None,
    ):
        r"""Base class for masked vector field estimator wrapper to adapt it to a
        vector field estimator API.

        Args:
            original_estimator: The MaskedConditionalVectorFieldEstimator instance
                to wrap.
            fixed_condition_mask: 1D tensor indicating which nodes are
                observed/conditioned on (1) or latent/to be inferred (0).
            fixed_edge_mask: 2D tensor indicating dependencies among input
              variables
                - `True` (or `1`): An edge exists from the row variable to the
                     column variable
                - `False` (or `0`): No edge exists.
        """

        T, F = original_estimator.input_shape

        # Input checks for fixed_condition_mask
        if not isinstance(fixed_condition_mask, torch.Tensor):
            raise TypeError("fixed_condition_mask must be a torch.Tensor.")
        if fixed_condition_mask.dim() != 1:
            raise ValueError(
                f"fixed_condition_mask must be 1-dimensional, got shape "
                f"{fixed_condition_mask.shape}."
            )
        if fixed_condition_mask.shape[0] != T:
            raise ValueError(
                f"fixed_condition_mask must have length {T} "
                f"(matching input_shape[0]), got {fixed_condition_mask.shape[0]}."
            )
        if not torch.all((fixed_condition_mask == 0) | (fixed_condition_mask == 1)):
            raise ValueError(
                "fixed_condition_mask must be boolean (0 or 1, True or False) "
                "for all entries."
            )

        # Input checks for fixed_edge_mask
        if not isinstance(fixed_edge_mask, torch.Tensor):
            raise TypeError("fixed_edge_mask must be a torch.Tensor.")
        if fixed_edge_mask.dim() != 2:
            raise ValueError(
                f"fixed_edge_mask must be 2-dimensional, got shape "
                f"{fixed_edge_mask.shape}."
            )
        if fixed_edge_mask.shape[0] != T or fixed_edge_mask.shape[1] != T:
            raise ValueError(
                f"fixed_edge_mask must have shape ({T}, {T}), got "
                f"{fixed_edge_mask.shape}."
            )
        if not torch.all((fixed_edge_mask == 0) | (fixed_edge_mask == 1)):
            raise ValueError(
                "fixed_edge_mask must be boolean (0 or 1, True or False) "
                "for all entries."
            )

        num_latent = int(torch.sum(fixed_condition_mask == 0).item())
        num_observed = int(torch.sum(fixed_condition_mask == 1).item())

        # Count number of latent and observed nodes
        self._new_input_shape = torch.Size((num_latent * F,))
        self._new_condition_shape = torch.Size((num_observed * F,))

        super().__init__(
            net=original_estimator.net,
            input_shape=self._new_input_shape,
            condition_shape=self._new_condition_shape,
            t_min=original_estimator.t_min,
            t_max=original_estimator.t_max,
        )

        self.SCORE_DEFINED = original_estimator.SCORE_DEFINED
        self.SDE_DEFINED = original_estimator.SDE_DEFINED
        self.MARGINALS_DEFINED = original_estimator.MARGINALS_DEFINED

        self._original_T = T
        self._original_F = F
        self._num_latent = num_latent
        self._num_observed = num_observed

        self._original_estimator = original_estimator

        # Move masks to the correct device and register them as buffers.
        self.register_buffer(
            "_fixed_condition_mask",
            fixed_condition_mask.clone().detach(),
        )
        self.register_buffer("_fixed_edge_mask", fixed_edge_mask.clone().detach())

        # Extract indices for latent (0) and observed (1) nodes
        # from the fixed_condition_mask
        self._latent_idx = (self._fixed_condition_mask == 0).nonzero(as_tuple=True)[0]
        self._observed_idx = (self._fixed_condition_mask == 1).nonzero(as_tuple=True)[0]

        # Get the mean/std for the latent nodes from the original estimator
        latent_mean_base_unflattened = original_estimator.mean_base[
            :, self._latent_idx, :
        ]
        latent_std_base_unflattened = original_estimator.std_base[
            :, self._latent_idx, :
        ]

        latent_mean_base_flattened = latent_mean_base_unflattened.flatten(start_dim=1)
        latent_std_base_flattened = latent_std_base_unflattened.flatten(start_dim=1)

        # Register these flattened buffers
        self.register_buffer("_mean_base", latent_mean_base_flattened.clone().detach())
        self.register_buffer("_std_base", latent_std_base_flattened.clone().detach())

        device = next(original_estimator.net.parameters()).device
        self.to(device)

    def forward(
        self, input: Tensor, condition: Tensor, time: Tensor, **kwargs
    ) -> Tensor:
        # Assemble full input from give input and condition
        # Take (..., T*F) and returns (..., T, F)

        # Check shapes are correct
        if input.shape[-1] != self._num_latent * self._original_F:
            raise ValueError(
                f"The last dimension of the input tensor must match the size of the "
                f"latent variables ({self._num_latent}*{self._original_F}), but got "
                f"{input.shape[-1]}. The wrapper expects a flattened tensor of shape "
                f"(..., num_latent * num_features)."
            )

        if condition.shape[-1] != self._num_observed * self._original_F:
            raise ValueError(
                f"The last dimension of the condition tensor must match the size of "
                f"the observed variables ({self._num_observed}*{self._original_F}), "
                f"but got {condition.shape[-1]}. The wrapper expects a flattened "
                f"tensor of shape (..., num_observed * num_features)."
            )

        total_elements = self._original_T * self._original_F
        if input.shape[-1] + condition.shape[-1] != total_elements:
            raise ValueError(
                f"The combined size of the last dimension of input and condition "
                f"({input.shape[-1] + condition.shape[-1]}) does not match the "
                f"total expected size of the original unmasked input "
                f"({total_elements}). Please ensure input and condition shapes are "
                f"correct."
            )

        if input.dim() < 2:
            raise ValueError(
                f"Input tensor must have at least 2 dimensions (batch, features), "
                f"but got {input.dim()} dimensions."
            )
        if condition.dim() < 2:
            raise ValueError(
                f"Condition tensor must have at least 2 dimensions (batch, features), "
                f"but got {condition.dim()} dimensions."
            )

        # Assemble full input from given input and condition
        full_inputs_tensor = self._assemble_full_inputs(input, condition)

        B = full_inputs_tensor.shape[0]
        expanded_cond_mask = self._fixed_condition_mask.unsqueeze(0).expand(B, -1)
        expanded_edge_mask = self._fixed_edge_mask.unsqueeze(0).expand(B, -1, -1)

        # Call the original masked estimator's forward method
        full_outputs = self._original_estimator.forward(
            input=full_inputs_tensor,
            time=time,
            condition_mask=expanded_cond_mask,
            edge_mask=expanded_edge_mask,
            **kwargs,
        )

        # Take B, T, F and return (B, num_latent*F) and (B, num_observed*F)
        latent_out = self._disassemble_full_outputs(full_outputs, input)
        return latent_out

    def loss(
        self,
        input: Tensor,
        condition: Tensor,
        **kwargs,
    ) -> Tensor:
        raise NotImplementedError(
            "The loss method of the MaskedConditionalVectorFieldEstimator "
            "is not intended to be used directly. If you want to use "
            "this estimator for a different inference method, "
            "please use the original masked estimator "
            "or implement a suitable loss."
        )

    # -------------------------- ODE METHODS --------------------------

    def ode_fn(self, input: Tensor, condition: Tensor, times: Tensor) -> Tensor:
        full_inputs_tensor = self._assemble_full_inputs(input, condition)  # (B, T, F)
        B = full_inputs_tensor.shape[0]
        expanded_cond_mask = self._fixed_condition_mask.unsqueeze(0).expand(B, -1)
        expanded_edge_mask = self._fixed_edge_mask.unsqueeze(0).expand(B, -1, -1)

        # original_estimator.ode_fn returns (B, T, F)
        full_outputs_ode = self._original_estimator.ode_fn(
            full_inputs_tensor,
            times,
            expanded_cond_mask,
            expanded_edge_mask,
        )
        # Disassemble and flatten the output
        latent_out = self._disassemble_full_outputs(
            full_outputs_ode, input
        )  # Returns (B, num_latent*F)
        return latent_out

    # -------------------------- SDE METHODS --------------------------

    def score(self, input: Tensor, condition: Tensor, t: Tensor) -> Tensor:
        # Adjust input over condition num trials
        if input.shape[1] == 1:
            input = input.repeat(1, condition.shape[0], 1)

        # Assemble full input from give input and condition
        # input: (B, num_latent * F), condition: (B, num_observed * F)
        full_inputs_tensor = self._assemble_full_inputs(input, condition)

        # Call the original estimator's loss
        B = full_inputs_tensor.shape[0]
        expanded_cond_mask = self._fixed_condition_mask.unsqueeze(0).expand(B, -1)
        expanded_edge_mask = self._fixed_edge_mask.unsqueeze(0).expand(B, -1, -1)

        full_score_outputs = self._original_estimator.score(
            full_inputs_tensor,
            t,
            expanded_cond_mask,
            expanded_edge_mask,
        )

        # Take B, T, F and return (B, num_latent*F) and (B, num_observed*F)
        latent_score = self._disassemble_full_outputs(full_score_outputs, input)
        # Returns (B, num_latent * F)
        return latent_score

    def mean_t_fn(self, times: Tensor) -> Tensor:
        return self._original_estimator.mean_t_fn(times)

    def std_fn(self, times: Tensor) -> Tensor:
        return self._original_estimator.std_fn(times)

    def drift_fn(self, input: Tensor, times: Tensor) -> Tensor:
        return self._original_estimator.drift_fn(input, times)

    def diffusion_fn(self, input: Tensor, times: Tensor) -> Tensor:
        return self._original_estimator.diffusion_fn(input, times)

    # ------------------------- UTILITIES ------------------------------

    def _assemble_full_inputs(self, input, condition):
        # Get batch shape and feature dimension
        B = int(torch.prod(torch.tensor(input.shape[:-1])).item())
        C = int(torch.prod(torch.tensor(condition.shape[:-1])).item())

        input_part_unflattened = input.reshape(B, self._num_latent, self._original_F)
        condition_part_unflattened = condition.reshape(
            -1, self._num_observed, self._original_F
        ).repeat(B // C, 1, 1)

        full_inputs = torch.zeros(
            B,
            self._original_T,
            self._original_F,
            dtype=input.dtype,
            device=input.device,
        )
        # Place unflattened parts into the correct positions
        full_inputs[:, self._latent_idx, :] = input_part_unflattened
        full_inputs[:, self._observed_idx, :] = condition_part_unflattened

        return full_inputs

    def _disassemble_full_outputs(self, full_outputs, original_latent_tensor):
        latent_part = full_outputs[:, self._latent_idx, :]  # (B, num_latent, F)

        return latent_part.reshape_as(original_latent_tensor)


class UnconditionalEstimator(nn.Module, ABC):
    r"""Base class for unconditional estimators that estimate properties of
    distributions without conditioning on an input.

    For example, this can be:
    - A density estimator of the pdf $p(x)$.
    - A vector field estimator e.g. $\nabla_x \log p(x)$.

    Subclasses of UnconditionalEstimator should implement the ``loss(input)``
    method to be compatible with sbi's training procedures.
    """

    def __init__(self, input_shape: Tuple) -> None:
        r"""Construct an unconditional estimator given shapes.

        Args:
            input_shape: Event shape of the input at which the density is being
                evaluated (and which is also the event_shape of samples).
        """
        super().__init__()
        self._input_shape = torch.Size(input_shape)

    @property
    def input_shape(self) -> torch.Size:
        r"""Return the input shape."""
        return self._input_shape

    @abstractmethod
    def loss(self, input: Tensor, **kwargs) -> Tensor:
        r"""Return the loss for training the estimator.

        Args:
            input: Inputs to evaluate the loss on of shape
                `(batch_dim, *input_event_shape)`.

        Returns:
            Loss of shape (batch_dim,)
        """
        pass

    def _check_input_shape(self, input: Tensor):
        r"""This method checks whether the input has the correct shape.

        Args:
            input: Inputs to evaluate the log probability on of shape
                    `(sample_dim_input, batch_dim_input, *event_shape_input)`.

        Raises:
            ValueError: If the input has a dimensionality that does not match
                        the expected input dimensionality.
            ValueError: If the shape of the input does not match the expected
                        input dimensionality.
        """
        input_shape = input.shape
        exp_input_shape = self.input_shape
        if len(input_shape) < len(exp_input_shape):
            raise ValueError(
                "Dimensionality of input is too small and does not match the "
                f"expected dimensionality {len(exp_input_shape)}. It should "
                f"be compatible with the provided input_shape {exp_input_shape}."
            )
        else:
            input_shape = input.shape[-len(self.input_shape) :]
            if input_shape != exp_input_shape:
                raise ValueError(
                    f"Shape of input {input_shape} does not match the "
                    f"expected input dimensionality {exp_input_shape}, as "
                    "provided by input_shape. Please reshape it accordingly."
                )


class UnconditionalDensityEstimator(UnconditionalEstimator):
    r"""Base class for unconditional density estimators.

    The density estimator class is a wrapper around neural networks that allows to
    evaluate the `log_prob`, `sample`, and provide the `loss` on $x$ values.

    Note:
        We assume that the input to the density estimator is a tensor of shape
        (sample_dim, batch_dim, *input_shape), where input_shape is the dimensionality
        of the input.

    """

    def __init__(self, net: nn.Module, input_shape: torch.Size) -> None:
        r"""Base class for density estimators.

        Args:
            net: Neural network or any parameterized model that is used to estimate the
                probability density of the input.
            input_shape: Event shape of the input at which the density is being
                evaluated (and which is also the event_shape of samples).
        """
        super().__init__(input_shape)
        self.net = net

    def log_prob(self, x: Tensor) -> Tensor:
        r"""Return the log probabilities of the inputs

        Args:
            x: Inputs to evaluate the log probability on of shape
                `(sample_dim_input, batch_dim_input, *event_shape_input)`.

        Returns:
            Sample-wise log probabilities.
        """

        self._neural_net.eval()
        return self._neural_net.log_prob(x)

    def sample(self, sample_shape: torch.Size) -> Tensor:
        r"""Return samples from the density estimator.

        Args:
            sample_shape: Shape of the samples to return.

        Returns:
            Samples of shape (*sample_shape, batch_dim, *event_shape_input).
        """

        return self._neural_net.sample(sample_shape)

    def sample_and_log_prob(self, sample_shape: torch.Size) -> Tuple[Tensor, Tensor]:
        r"""Return samples and their density from the density estimator.

        Args:
            sample_shape: Shape of the samples to return.

        Returns:
            Samples and associated log probabilities.

        Note:
            For some density estimators, computing log_probs for samples is
            more efficient than computing them separately. This method should
            then be overwritten to provide a more efficient implementation.
        """

        samples = self.sample(sample_shape)
        log_probs = self.log_prob(samples)
        return samples, log_probs
