# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Various PyTorch utility functions."""

import os
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    OrderedDict,
    Tuple,
    Union,
)

import numpy as np
import torch
from torch import Tensor, float32
from torch.distributions import Independent, Uniform
from torch.nn import Module

from sbi.sbi_types import Array, OneOrMore
from sbi.utils.typechecks import is_nonnegative_int, is_positive_int

_HAS_WARNED_MPS_FALLBACK: bool = False


def set_validate_args(
    dist: torch.distributions.Distribution, validate_args: bool
) -> None:
    """Set argument and sample validation on this distribution instance only.

    Torch only offers `set_default_validate_args`, which mutates the default for every
    distribution constructed afterwards, globally. Validation runs at every layer of a
    composed distribution, so this recurses into any nested distribution, including
    private ones such as `Beta._dirichlet`.

    Args:
        dist: The distribution instance to configure.
        validate_args: Whether the instance validates arguments and samples.
    """
    dist._validate_args = validate_args
    for value in vars(dist).values():
        members = value if isinstance(value, (list, tuple)) else [value]
        for member in members:
            if isinstance(member, torch.distributions.Distribution):
                set_validate_args(member, validate_args)


def process_device(device: Optional[Union[str, torch.device]]) -> str:
    """Resolve `device` to a canonical device string and set torch up to use it.

    All spellings of the same device are treated identically, so `"gpu"`, `"mps"`,
    `"mps:0"` and `torch.device("mps")` are interchangeable on Apple silicon.

    Args:
        device: target torch device. Either `"gpu"`, which selects CUDA if available
            and MPS otherwise, `None` for the CPU, or an explicit device such as
            `"cpu"`, `"cuda"`, `"cuda:1"`, `"mps"`, or a `torch.device`.

    Returns:
        Canonical device string, i.e. `"cpu"`, `"cuda:{index}"` or `"mps:0"`.

    Raises:
        RuntimeError: if `"gpu"` is requested without a GPU backend, if `device`
            cannot be interpreted, or if it cannot hold a tensor.
    """
    if device is None:
        return "cpu"

    # If the user just passes 'gpu', search for CUDA or MPS.
    if device == "gpu":
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
        elif torch.backends.mps.is_available():
            device = "mps:0"
        else:
            raise RuntimeError(
                "Neither CUDA nor MPS is available. "
                "Please make sure to install a version of PyTorch that supports "
                "CUDA or MPS."
            )

    device = canonical_device(device)

    if device == "cpu":
        # `check_device` allocates via `torch.randn`, which would consume the CPU
        # RNG and perturb seeded runs.
        return "cpu"

    if device.startswith("mps"):
        _warn_if_mps_fallback_disabled()
        # MPS framework does not support double precision, and `check_device`
        # below allocates a tensor.
        torch.set_default_dtype(torch.float32)

    check_device(device)

    return device


def canonical_device(device: Union[str, torch.device]) -> str:
    """Return `device` as a string carrying an explicit index for GPU devices.

    Device strings can only be compared once the index is resolved, because
    `torch.device("cuda")` and `torch.device("cuda:0")` are not equal even though
    both refer to the same device.

    Args:
        device: any torch device or device string, e.g. `"cuda"` or `"mps:0"`.

    Returns:
        Device string, e.g. `"cpu"`, `"cuda:0"` or `"mps:0"`.

    Raises:
        RuntimeError: if `device` cannot be interpreted as a torch device.
    """
    try:
        resolved = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise RuntimeError(
            f"Could not interpret '{device}' as a torch device. It should be "
            "something like 'cpu', 'cuda', 'cuda:0', or 'mps'."
        ) from exc

    if resolved.type == "cpu":
        return "cpu"

    if resolved.index is None:
        if resolved.type == "cuda":
            index = torch.cuda.current_device() if torch.cuda.is_available() else 0
            return f"cuda:{index}"
        if resolved.type == "mps":
            return "mps:0"

    return str(resolved)


def mps_fallback_enabled() -> bool:
    """Whether PyTorch falls back to the CPU for operators missing on MPS.

    Returns:
        Whether `PYTORCH_ENABLE_MPS_FALLBACK` is set to a non-zero value. Note that
        PyTorch only honors it if it was set before torch was imported.
    """
    return os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0").strip() not in ("", "0")


def _warn_if_mps_fallback_disabled() -> None:
    """Warn once per process that operators missing on MPS will raise."""
    global _HAS_WARNED_MPS_FALLBACK
    if not _HAS_WARNED_MPS_FALLBACK and not mps_fallback_enabled():
        _HAS_WARNED_MPS_FALLBACK = True
        warnings.warn(
            "Using the MPS backend, for which PyTorch does not implement all "
            "operations. To fall back to the CPU for those instead of raising, set "
            "the environment variable PYTORCH_ENABLE_MPS_FALLBACK=1 before importing "
            "torch. PyTorch registers the fallback at import time, so sbi cannot "
            "enable it for you.",
            UserWarning,
            stacklevel=3,
        )


def gpu_available() -> bool:
    """Check whether GPU is available."""
    return torch.cuda.is_available() or torch.backends.mps.is_available()


def check_device(device: Union[str, torch.device]) -> None:
    """Check whether the device is valid.

    Args:
        device: target torch device
    """
    try:
        torch.randn(1, device=device)
    except (RuntimeError, AssertionError) as exc:
        raise RuntimeError(
            f"""Could not instantiate torch.randn(1, device={device}). Make sure
             the device is set up properly and that you are passing the
             corresponding device string. It should be something like 'cuda',
             'cuda:0', or 'mps'. Error message: {exc}."""
        ) from exc


def check_if_prior_on_device(
    device: Union[str, torch.device], prior: Optional[Any] = None
) -> None:
    """Try to sample from the prior, and check that the returned data is on the correct
    trainin device. If the prior is `None`, simplys pass.

    Args:
        device: target torch training device
        prior: any simulator outputing torch `Tensor`
    """
    if prior is None:
        pass
    else:
        prior_device = prior.sample((1,)).device
        training_device = torch.zeros(1, device=device).device
        assert prior_device == training_device, (
            f"Prior device '{prior_device}' must match training device "
            f"'{training_device}'. When training on GPU make sure to "
            "pass a prior initialized on the GPU as well, e.g., "
            "use `.to(device)` for sbi priors or "
            "prior = torch.distributions.Normal"
            "(torch.zeros(2, device='cuda'), scale=1.0)`, or ."
        )


def infer_module_device(module: torch.nn.Module, fallback: str) -> str:
    """Infer device from module parameters or buffers, falling back to `fallback`.

    Args:
        module: The module to inspect.
        fallback: Device string returned (with a warning) if the module has no
            parameters or buffers.

    Returns:
        Device string, e.g. ``"cpu"`` or ``"cuda:0"``.
    """
    try:
        return str(next(module.parameters()).device)
    except StopIteration:
        try:
            return str(next(module.buffers()).device)
        except StopIteration:
            warnings.warn(
                f"{type(module).__name__} has no parameters/buffers; "
                f"falling back to device='{fallback}'.",
                stacklevel=2,
            )
            return fallback


def infer_tensor_device(obj: Any) -> Optional[str]:
    """Return the device of the first tensor or module parameter reachable from `obj`.

    Args:
        obj: Root object of the graph to inspect.

    Returns:
        Device string, e.g. `"cpu"`, `"cuda:0"` or `"mps:0"`, or `None` if the graph
        contains no tensor or module with parameters or buffers.
    """

    def _infer(current: Any, seen: set) -> Optional[str]:
        if id(current) in seen:
            return None
        seen.add(id(current))

        if isinstance(current, Module):
            try:
                return str(next(current.parameters()).device)
            except StopIteration:
                try:
                    return str(next(current.buffers()).device)
                except StopIteration:
                    pass

        if isinstance(current, Tensor):
            return str(current.device)

        if isinstance(current, dict):
            for value in current.values():
                device = _infer(value, seen)
                if device is not None:
                    return device
        elif isinstance(current, (list, tuple)):
            for value in current:
                device = _infer(value, seen)
                if device is not None:
                    return device
        elif hasattr(current, "__dict__"):
            for value in vars(current).values():
                device = _infer(value, seen)
                if device is not None:
                    return device

        return None

    return _infer(obj, set())


def tile(x: Tensor, n: int) -> Tensor:
    """Tiles a tensor `x` by repeating it `n` times along a new leading dimension.

    Args:
        x: Input tensor to tile.
        n: Number of times to tile the tensor.

    Returns:
        Tiled tensor.
    """

    if not is_positive_int(n):
        raise TypeError("Argument `n` must be a positive integer.")
    x_ = x.reshape(-1)
    x_ = x_.repeat(n)
    x_ = x_.reshape(n, -1)
    x_ = x_.transpose(1, 0)
    x_ = x_.reshape(-1)
    return x_


def sum_except_batch(x: Tensor, num_batch_dims: int = 1) -> Tensor:
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions.

    Args:
        x: Input tensor.
        num_batch_dims: Number of batch dimensions to keep. Defaults to 1.

    Returns:
        Tensor with all non-batch dimensions summed.
    """

    if not is_nonnegative_int(num_batch_dims):
        raise TypeError("Number of batch dimensions must be a non-negative integer.")
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


def split_leading_dim(x: Tensor, shape: List[int]) -> Tensor:
    """Reshapes the leading dim of `x` to have the given shape.

    Args:
        x: Input tensor.
        shape: Desired shape for the leading dimension.

    Returns:
        Tensor with reshaped leading dimension.
    """
    new_shape = torch.Size(shape) + x.shape[1:]
    return torch.reshape(x, new_shape)


def merge_leading_dims(x: Tensor, num_dims: int) -> Tensor:
    """Reshapes the tensor `x` such that the first `num_dims` dimensions are merged
    to one.

    Args:
        x: Input tensor.
        num_dims: Number of leading dimensions to merge.

    Returns:
        Tensor with first `num_dims` dimensions merged into one.
    """
    if not is_positive_int(num_dims):
        raise TypeError("Number of leading dims must be a positive integer.")
    if num_dims > x.dim():
        raise ValueError(
            "Number of leading dims can't be greater than total number of dims."
        )
    new_shape = torch.Size([-1]) + x.shape[num_dims:]
    return torch.reshape(x, new_shape)


def repeat_rows(x: Tensor, num_reps: int) -> Tensor:
    """Each row of tensor `x` is repeated `num_reps` times along leading dimension.

    Args:
        x: Input tensor.
        num_reps: Number of times to repeat each row.

    Returns:
        Tensor with each row repeated `num_reps` times.
    """

    if not is_positive_int(num_reps):
        raise TypeError("Number of repetitions must be a positive integer.")
    shape = x.shape
    x = x.unsqueeze(1)
    x = x.expand(shape[0], num_reps, *shape[1:])
    return merge_leading_dims(x, num_dims=2)


def tensor2numpy(x: Tensor) -> np.ndarray:
    """Converts a PyTorch tensor to a NumPy array.

    Detaches the tensor from the computation graph and moves it to CPU
    before converting.

    Args:
        x: Input tensor.

    Returns:
        NumPy array with the same data as `x`.
    """
    return x.detach().cpu().numpy()


def logabsdet(x: Tensor) -> Tensor:
    """Returns the log absolute determinant of square matrix x.

    Args:
        x: Square matrix tensor.

    Returns:
        Scalar tensor containing the log absolute determinant.
    """
    # Note: torch.logdet() only works for positive determinant.
    _, res = torch.slogdet(x)
    return res


def random_orthogonal(size: int) -> Tensor:
    """Returns a random orthogonal matrix as a 2-dim tensor of shape [size, size].

    Uses the QR decomposition of a random Gaussian matrix to generate
    a uniformly distributed orthogonal matrix.

    Args:
        size: Dimension of the square orthogonal matrix.

    Returns:
        Random orthogonal matrix of shape (size, size).
    """

    # Use the QR decomposition of a random Gaussian matrix.
    x = torch.randn(size, size)
    q, _ = torch.linalg.qr(x)
    return q


def get_num_parameters(model: Module) -> int:
    """Returns the number of trainable parameters in a model of type nets.Module.

    Args:
        model: PyTorch module containing trainable parameters.

    Returns:
        Total number of trainable parameters.
    """
    num_parameters = 0
    for parameter in model.parameters():
        num_parameters += torch.numel(parameter)
    return num_parameters


def create_alternating_binary_mask(features: int, even: bool = True) -> Tensor:
    """Creates a binary mask of a given dimension which alternates its masking.

    Args:
        features: Dimension of mask.
        even: If True, even indices are assigned 1s and odd indices 0s.
            If False, vice versa. Defaults to True.

    Returns:
        Alternating binary mask of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    start = 0 if even else 1
    mask[start::2] += 1
    return mask


def create_mid_split_binary_mask(features: int) -> Tensor:
    """Creates a binary mask of a given dimension which splits its masking
    at the midpoint

    Args:
        features: Dimension of mask.

    Returns:
        Binary mask split at midpoint of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    midpoint = features // 2 if features % 2 == 0 else features // 2 + 1
    mask[:midpoint] += 1
    return mask


def create_random_binary_mask(features: int) -> Tensor:
    """Creates a random binary mask of a given dimension with half of its entries
    randomly set to 1s.

    Args:
        features: Dimension of mask.

    Returns:
        Binary mask with half of its entries set to 1s, of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    weights = torch.ones(features).float()
    num_samples = features // 2 if features % 2 == 0 else features // 2 + 1
    indices = torch.multinomial(
        input=weights, num_samples=num_samples, replacement=False
    )
    mask[indices] += 1
    return mask


def searchsorted(bin_locations: Tensor, inputs: Tensor, eps: float = 1e-6) -> Tensor:
    """Finds the indices of the bins to which each input value belongs.

    Args:
        bin_locations: Tensor of bin edges.
        inputs: Tensor of values to search for in the bins.
        eps: Small value added to the last bin edge to ensure correct boundary
            behavior. Defaults to 1e-6.

    Returns:
        Tensor of bin indices for each input value.
    """

    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def cbrt(x: Tensor) -> Tensor:
    """Cube root. Equivalent to torch.pow(x, 1/3), but numerically stable.

    Args:
        x: Input tensor.

    Returns:
        Element-wise cube root of `x`.
    """
    return torch.sign(x) * torch.exp(torch.log(torch.abs(x)) / 3.0)


def get_temperature(max_value: float, bound: float = 1 - 1e-3) -> Tensor:
    """Returns the temperature such that sigmoid(temperature * max_value) = bound.

    If the computed temperature is greater than 1, returns 1.

    Args:
        max_value: Maximum value of the dataset.
        bound: Target value for sigmoid(temperature * max_value).
            Defaults to 1 - 1e-3.

    Returns:
        Scalar tensor containing the temperature value, capped at 1.
    """
    max_value_t = torch.Tensor([max_value])
    bound_t = torch.Tensor([bound])
    temperature = torch.min(
        -(1 / max_value_t) * (torch.log1p(-bound_t) - torch.log(bound_t)),
        torch.ones(1),
    )
    return temperature


def gaussian_kde_log_eval(samples: Tensor, query: Tensor) -> Tensor:
    """Evaluates the log probability of query points under a Gaussian KDE.

    Fits a Gaussian kernel density estimator to `samples` using
    Silverman's rule of thumb for bandwidth selection, then evaluates
    the log probability at each point in `query`.

    Args:
        samples: Tensor of shape (N, D) used to fit the KDE.
        query: Tensor of shape (..., D) of points to evaluate.

    Returns:
        Tensor of log probabilities for each query point.
    """
    N, D = samples.shape[0], samples.shape[-1]
    std = N ** (-1 / (D + 4))
    precision = (1 / (std**2)) * torch.eye(D)
    a = query - samples
    b = a @ precision
    c = -0.5 * torch.sum(a * b, dim=-1)
    d = -np.log(N) - (D / 2) * np.log(2 * np.pi) - D * np.log(std)
    c += d
    return torch.logsumexp(c, dim=-1)


class BoxUniform(Independent):
    """Uniform distribution in multiple dimensions."""

    def __init__(
        self,
        low: Union[Tensor, Array],
        high: Union[Tensor, Array],
        reinterpreted_batch_ndims: int = 1,
        device: Optional[str] = None,
    ):
        """Multidimensional uniform distribution defined on a box.

        A :class:`~torch.distributions.uniform.Uniform` distribution initialized \
        with e.g. a parameter vector low or high of length 3 will result \
        in a *batch* dimension of length 3. A log_prob evaluation will then \
        output three numbers, one for each of the independent Uniforms in the \
        batch. Instead, a :class:`BoxUniform` initialized in the same way has three \
        *event* dimensions, and returns a scalar log_prob corresponding to whether \
        the evaluated point is in the box defined by low and high or outside.

        Refer to :class:`~torch.distributions.uniform.Uniform`\
        and :class:`~torch.distributions.independent.Independent` for \
        further documentation.

        Args:
            low: lower range (inclusive).
            high: upper range (exclusive).
            reinterpreted_batch_ndims (int): the number of batch dims to \
            reinterpret as event dims.
            device (Optional): device of the prior, inferred from low arg, \
            defaults to "cpu", should match the training device when used in SBI.

        Example:
        --------

        ::

            import torch
            from sbi.utils.torchutils import BoxUniform

            # Define lower bounds
            low = torch.tensor([0.0, 0.0, 0.0])

            # Define upper bounds
            high = torch.tensor([1.0, 1.0, 1.0])

            box_uniform = BoxUniform(low, high)

            # Sample from the box_uniform
            N_samples = 100
            sample = box_uniform.sample((N_samples,))

            # Evaluate the log probability of the sample
            log_prob = box_uniform.log_prob(sample)
        """

        # Type checks.
        assert isinstance(low, Tensor) and isinstance(high, Tensor), (
            f"low and high must be tensors but are {type(low)} and {type(high)}."
        )
        if not low.device == high.device:
            raise RuntimeError(
                "Expected all tensors to be on the same device, but found at least"
                f"two devices, {low.device} and {high.device}."
            )

        # Device handling
        device = str(low.device) if device is None else device
        device = process_device(device)
        self.device = device
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

        self.low = torch.as_tensor(low, dtype=torch.float32, device=device)
        self.high = torch.as_tensor(high, dtype=torch.float32, device=device)

        super().__init__(
            Uniform(
                low=torch.as_tensor(
                    low, dtype=torch.float32, device=torch.device(device)
                ),
                high=torch.as_tensor(
                    high, dtype=torch.float32, device=torch.device(device)
                ),
                validate_args=False,
            ),
            reinterpreted_batch_ndims,
            validate_args=False,
        )

    def to(self, device: Union[str, torch.device]) -> None:
        """
        Moves the distribution to the specified device **in place**.

        Args:
            device: Target device (e.g., "cpu", "cuda", "mps").

        Example:
        --------

        ::

            device = "cuda"
            prior = BoxUniform(low=torch.zeros(2), high=torch.ones(2))
            prior.to(device) #inplace
        """
        # Update the device attribute
        device = process_device(device)
        self.device = device

        # Move tensors to the new device
        self.low = self.low.to(device=device)
        self.high = self.high.to(device=device)

        # Rebuilding below would otherwise reset this to torch's class default.
        validate_args = self._validate_args

        super().__init__(
            Uniform(
                low=self.low,
                high=self.high,
                validate_args=validate_args,
            ),
            self.reinterpreted_batch_ndims,
            validate_args=validate_args,
        )


def ensure_theta_batched(theta: Tensor) -> Tensor:
    r"""
    Return parameter set theta that has a batch dimension, i.e. has shape
     (1, shape_of_single_theta)

     Args:
         theta: parameters $\theta$, of shape (n) or (1,n)
     Returns:
         Batched parameter set $\theta$
    """

    # => ensure theta has shape (1, dim_parameter)
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)

    return theta


def ensure_x_batched(x: Tensor) -> Tensor:
    """
    Return simulation output x that has a batch dimension, i.e. has shape
    (1, shape_of_single_x).

    Args:
         x: simulation output of shape (n) or (1,n).
     Returns:
         Batched simulation output x.
    """

    # ensure x has shape (1, shape_of_single_x). If shape[0] > 1, we assume that
    # the batch-dimension is missing, even though ndim might be >1 (e.g. for images)
    if x.shape[0] > 1 or x.ndim == 1:
        x = x.unsqueeze(0)

    return x


def atleast_2d_many(*arys: Array) -> OneOrMore[Tensor]:
    """Return tensors with at least dimension 2.

    Tensors or arrays of dimension 0 or 1 will get additional dimension(s) prepended.

    Returns:
        Tensor or list of tensors all with dimension >= 2.
    """
    if len(arys) == 1:
        arr = arys[0]
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr)
        return atleast_2d(arr)
    else:
        return [atleast_2d_many(arr) for arr in arys]  # type: ignore


def atleast_2d(t: Tensor) -> Tensor:
    return t if t.ndim >= 2 else t.reshape(1, -1)


def maybe_add_batch_dim_to_size(s: torch.Size) -> torch.Size:
    """
    Take a torch.Size and add a batch dimension to it if dimensionality of size is 1.

    (N) -> (1,N)
    (1,N) -> (1,N)
    (N,M) -> (N,M)
    (1,N,M) -> (1,N,M)

    Args:
        s: Input size, possibly without batch dimension.

    Returns: Batch size.

    """
    return s if len(s) >= 2 else torch.Size([1]) + s  # type: ignore


def atleast_2d_float32_tensor(arr: Union[Tensor, np.ndarray]) -> Tensor:
    return atleast_2d(torch.as_tensor(arr, dtype=float32))


def batched_first_of_batch(t: Tensor) -> Tensor:
    """
    Takes in a tensor of shape (N, M) and outputs tensor of shape (1,M).
    """
    return t[:1]


def assert_all_finite(quantity: Tensor, description: str = "tensor") -> None:
    """Raise if tensor quantity contains any NaN or Inf element."""

    msg = f"NaN/Inf present in {description}."
    if not torch.isfinite(quantity).all():
        raise ValueError(msg)


def assert_not_nan_or_plus_inf(quantity: Tensor, description: str = "tensor") -> None:
    """Raise if tensor quantity contains any NaN or +Inf element."""

    msg = f"NaN/ +Inf present in {description}."
    assert not (torch.isposinf(quantity).any()) and not (torch.isnan(quantity).any()), (
        msg
    )


def _base_recursor(
    obj: object,
    parent: Optional[object] = None,
    key: Optional[str] = None,
    check: Callable[..., bool] = lambda obj: False,
    action: Callable[..., object] = lambda obj: obj,
    _active: Optional[set[int]] = None,
):
    """Recursive function that traverses objects (e.g. Distributions) and applies
    an action to any encountered object that passes the check.

    Used e.g. to move all tensors within a distribution to a given device.

    Args:
        obj: An object which serves as root of the traversal.
        parent: The previously traversed object.
        key: The name of the previously traversed object.
        check: A function that inputs an object and outputs a boolean.
            If the check evaluates to True, then ``action`` is applied.
        action: A function that specifies an operation on an object and returns
            a modified version.
        _active: Object identities on the active recursion path. Used internally
            to avoid infinite recursion in cyclic object graphs.
    """
    if _active is None:
        _active = set()

    obj_id = id(obj)
    if obj_id in _active:
        return
    _active.add(obj_id)

    if isinstance(obj, Module) and check(obj):
        action(obj)
    elif isinstance(obj, (Dict, OrderedDict)):
        for k, o in obj.items():
            if check(o):
                obj[k] = action(o)
            else:
                _base_recursor(
                    o,
                    parent=obj,
                    key=k,
                    check=check,
                    action=action,
                    _active=_active,
                )
    elif isinstance(obj, type):
        # Skip class/type objects to avoid modifying immutable C extension types
        # (e.g. torch.LongTensor) which fail on Python 3.13+.
        pass
    elif hasattr(obj, "__dict__"):
        for k, o in obj.__dict__.items():
            if check(o):
                setattr(obj, k, action(o))
            else:
                _base_recursor(
                    o,
                    parent=obj,
                    key=k,
                    check=check,
                    action=action,
                    _active=_active,
                )
    elif isinstance(obj, (List, Tuple, Generator)):
        new_obj = []
        for o in obj:
            if check(o):
                new_obj.append(action(o))
            else:
                _base_recursor(o, check=check, action=action, _active=_active)
                new_obj.append(o)
        if parent is not None and key is not None:
            setattr(parent, key, type(obj)(new_obj))  # type: ignore

    _active.remove(obj_id)


def move_all_tensor_to_device(obj: object, device: Union[str, torch.device]) -> None:
    """Recursively move all tensors and modules within an object to a device.

    Traverses the object's attributes, dictionaries, lists, and tuples,
    moving any encountered ``Tensor`` or ``Module`` to the specified device.

    Note:
        Leaf tensors with ``requires_grad=True`` cannot be moved in-place.
        A ``ValueError`` is raised if such a tensor is on the wrong device.

    Args:
        obj: The root object to traverse.
        device: The target device.
    """

    def check(o: object) -> bool:
        return isinstance(o, (Tensor, Module))

    def action(o: object) -> object:
        if isinstance(o, Tensor) and o.requires_grad and o.is_leaf:
            # Moving leaf tensors inplace is hard. Cant call .to as this would create a
            # copy and thus results in non-leaf tensors.
            if canonical_device(o.device) != canonical_device(device):
                raise ValueError(
                    f"Cannot move leaf tensor with requires_grad=True from "
                    f"{o.device} to {device}. Please initialize it on the "
                    f"correct device."
                )
            else:
                return o
        else:
            return o.to(device)  # type: ignore

    with torch.no_grad():
        _base_recursor(obj, check=check, action=action)
