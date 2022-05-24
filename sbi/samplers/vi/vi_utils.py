from copy import deepcopy
from typing import (
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    OrderedDict,
    Tuple,
    Union,
)

import numpy as np
import torch
from pyro.distributions import TransformedDistribution
from pyro.distributions.torch_transform import TransformModule
from torch import Tensor
from torch.distributions import Distribution
from torch.distributions.transforms import ComposeTransform, IndependentTransform
from torch.nn import Module

from sbi.types import TorchTransform


def filter_kwrags_for_func(f: Callable, kwargs: Dict) -> Dict:
    """This function will filter a dictionary of possible arguments for arguments the
    function can use.

    Args:
        f: Function for which kwargs are filtered
        kwargs: Possible kwargs for function

    Returns:
        dict: Subset of kwargs, which the function f can take as arguments.

    """
    args = f.__code__.co_varnames
    new_kwargs = dict([(key, val) for key, val in kwargs.items() if key in args])
    return new_kwargs


def get_parameters(t: Union[TorchTransform, TransformModule]) -> Iterable:
    """Recursive helper function which can be used to extract parameters from
    TransformedDistributions.

    Args:
        t: A TorchTransform object, which is scanned for the "parameters" attribute.

    Yields:
        Iterator[Iterable]: Generator of parameters.
    """
    if hasattr(t, "parameters"):
        yield from t.parameters()  # type: ignore
    elif isinstance(t, ComposeTransform):
        for part in t.parts:
            yield from get_parameters(part)
    elif isinstance(t, IndependentTransform):
        yield from get_parameters(t.base_transform)
    else:
        pass


def get_modules(t: Union[TorchTransform, TransformModule]) -> Iterable:
    """Recursive helper function which can be used to extract modules from
    TransformedDistributions.

    Args:
        t: A TorchTransform object, which is scanned for the "modules" attribute.

    Yields:
        Iterator[Iterable]: Generator of TransformModules
    """
    if isinstance(t, Module):
        yield t
    elif isinstance(t, ComposeTransform):
        for part in t.parts:
            yield from get_modules(part)
    elif isinstance(t, IndependentTransform):
        yield from get_modules(t.base_transform)
    else:
        pass


def check_parameters_modules_attribute(q: TransformedDistribution) -> None:
    """Checks a parameterized distribution object for valid `parameters` and `modules`.

    Args:
        q: Distribution object
    """

    if not hasattr(q, "parameters"):
        raise ValueError(
            """The variational distribution requires an `parameters` attribute, which
            returns an iterable of parameters"""
        )
    else:
        assert isinstance(q.parameters, Callable), "The parameters must be callable"
        parameters = q.parameters()
        assert isinstance(
            parameters, Iterable
        ), "The parameters return value must be iterable"
        trainable = 0
        for p in parameters:
            assert isinstance(p, torch.Tensor)
            if p.requires_grad:
                trainable += 1
        assert (
            trainable > 0
        ), """Nothing to train, atleast one of the parameters must have an enabled
            gradient."""
    if not hasattr(q, "modules"):
        raise ValueError(
            """The variational distribution requires an modules attribute, which returns
            an iterable of parameters."""
        )
    else:
        assert isinstance(q.modules, Callable), "The parameters must be callable"
        modules = q.modules()
        assert isinstance(
            modules, Iterable
        ), "The parameters return value must be iterable"
        for m in modules:
            assert isinstance(
                m, Module
            ), "The modules must contain PyTorch Module objects"


def check_sample_shape_and_support(q: Distribution, prior: Distribution) -> None:
    """Checks the samples shape and support between variational distribution and the
    prior. Especially it checks if the shapes match and that the support between q and
    the prior matches (a property which holds for the true posterior in any case).

    Args:
        q: Variational distribution which is checked
        prior: Prior to check certain attributes which should be satisfied.

    """
    assert (
        q.event_shape == prior.event_shape
    ), "The event shape of q must match that of the prior"
    assert (
        q.batch_shape == prior.batch_shape
    ), "The batch sahpe of q must match that of the prior"

    sample_shape = torch.Size((1000,))
    samples = q.sample(sample_shape)
    samples_prior = prior.sample(sample_shape).to(samples.device)
    try:
        _ = prior.support
        has_support = True
    except (NotImplementedError, AttributeError):
        has_support = False
    if has_support:
        assert all(
            prior.support.check(samples)  # type: ignore
        ), "The support of q must match that of the prior"
    assert (
        samples.shape == samples_prior.shape
    ), "Something is wrong with sample shape and event_shape or batch_shape attributes."
    assert torch.isfinite(
        q.log_prob(samples_prior)
    ).all(), "Invalid values in logprob on prior samples."
    assert torch.isfinite(
        prior.log_prob(samples)
    ).all(), "Invalid values in logprob on q samples."


def check_variational_distribution(q: Distribution, prior: Distribution) -> None:
    """Runs all basic checks such the q is `valid`.

    Args:
        q: Variational distribution which is checked
        prior: Prior to check certain attributes which should be satisfied.

    """
    check_parameters_modules_attribute(q)
    check_sample_shape_and_support(q, prior)


def add_parameters_module_attributes(
    q: Distribution, parameters: Callable, modules: Callable
):
    """Sets the attribute `parameters` and `modules` on q

    Args:
        q: Variational distribution which is checked
        parameters: A function which returns an iterable of leaf tensors.
        modules: A function which returns an iterable of nn.Module


    """
    setattr(q, "parameters", parameters)
    setattr(q, "modules", modules)


def add_parameter_attributes_to_transformed_distribution(
    q: TransformedDistribution,
) -> None:
    """A function that will add `parameters` and `modules` to q automatically, if q is a
    TransformedDistribution.

    Args:
        q: Variational distribution instances TransformedDistribution.


    """

    def parameters():
        """Returns the parameters of the distribution."""
        if hasattr(q.base_dist, "parameters"):
            yield from q.base_dist.parameters()
        for t in q.transforms:
            yield from get_parameters(t)

    def modules():
        """Returns the modules of the distribution."""
        if hasattr(q.base_dist, "modules"):
            yield from q.base_dist.modules()
        for t in q.transforms:
            yield from get_modules(t)

    add_parameters_module_attributes(q, parameters, modules)


def adapt_variational_distribution(
    q: TransformedDistribution,
    prior: Distribution,
    link_transform: Callable,
    parameters: Iterable = [],
    modules: Iterable = [],
) -> Distribution:
    """This will adapt a distribution to be compatible with DivergenceOptimizers.
    Especially it will make sure that the distribution has parameters and that it
    satisfies obvious contraints which a posterior must satisfy i.e. the support must be
    equal to that of the prior.

    Args:
        q: Variational distribution.
        prior: Prior distribution
        theta_transform: Theta transformation.
        parameters: List of parameters.
        modules: List of modules.

    Returns:
        TransformedDistribution: Compatible variational distribution.

    """
    # Extract user define parameters
    def parameters_fn():
        """Returns the parameters of the distribution."""
        return parameters

    def modules_fn():
        """Returns the modules of the distribution."""
        return modules

    if isinstance(q, TransformedDistribution):
        if parameters == [] or modules_fn == []:
            add_parameter_attributes_to_transformed_distribution(q)
        else:
            add_parameters_module_attributes(q, parameters_fn, modules_fn)
        if hasattr(prior, "support") and q.support != prior.support:
            q = TransformedDistribution(q.base_dist, q.transforms + [link_transform])
    else:
        if hasattr(prior, "support") and q.support != prior.support:
            q = TransformedDistribution(q, [link_transform])
        add_parameters_module_attributes(q, parameters_fn, modules_fn)

    return q


def _base_recursor(
    obj: object,
    parent: Optional[object] = None,
    key: Optional[str] = None,
    check: Callable[..., bool] = lambda obj: False,
    action: Callable[..., object] = lambda obj: obj,
):
    """This functions is a recursive function that traverses classes i.e. Distributions
    and checks any encountered object according to `check`. If a check evaluates to
    True, then an action is applied as specified in `action`. We use it e.g. to
    move tensors to a given device.

    Args:
        obj: An object which serves as root of the traversal.
        parent: The previously traversed object.
        key: The name of the previously traversed object
        check: A function that inputs a object which is currently investigates and
            outputs a boolean. If the check evalues to True, then `action` is applied.
        action: A function that specifies an operation on an object and return an
            modified version.
    """
    if isinstance(obj, Module) and check(obj):
        action(obj)
    elif isinstance(obj, Dict) or isinstance(obj, OrderedDict):
        for k, o in obj.items():
            if check(o):
                obj[k] = action(o)
            else:
                _base_recursor(o, parent=obj, key=k, check=check, action=action)
    elif hasattr(obj, "__dict__"):
        for k, o in obj.__dict__.items():
            if check(o):
                setattr(obj, k, action(o))
            else:
                _base_recursor(o, parent=obj, key=k, check=check, action=action)
    elif isinstance(obj, List) or isinstance(obj, Tuple) or isinstance(obj, Generator):
        new_obj = []
        for o in obj:
            if check(o):
                new_obj.append(action(o))
            else:
                _base_recursor(o, check=check, action=action)
                new_obj.append(o)
        if parent is not None and key is not None:
            setattr(parent, key, type(obj)(new_obj))  # type: ignore
    else:
        return


def detach_all_non_leaf_tensors(obj: object) -> None:
    """This detaches all non leaf tensors, which especially is required if one wants to
    create a deepcopy of the object. This is because PyTorch does not support the
    deepcopy protocol on non-leaf tensors.

    Args:
        obj: An object which is traversed for non_leaf tensors.

    """

    def check(o):
        return isinstance(o, Tensor) and o.requires_grad and not o.is_leaf

    def action(o):
        return o.detach()

    with torch.no_grad():
        _base_recursor(obj, check=check, action=action)


def move_all_tensor_to_device(obj, device):
    def check(o):
        return isinstance(o, Tensor) or isinstance(o, Module)

    def action(o):
        if isinstance(o, Tensor) and o.requires_grad and o.is_leaf:
            # Moving leaf tensors inplace is hard. Cant call .to as this would create a
            # copy and thus results in non-leaf tensors.
            if str(o.device) != str(device):
                print(o)
                raise ValueError(
                    "Some of your leaf tensors are on the wrong device, we cant move"
                    "them automatically please initialize them correctly. Move e.g. "
                    f"{o} from {o.device} to {device}"
                )
            else:
                return o
        else:
            return o.to(device)

    with torch.no_grad():
        _base_recursor(obj, check=check, action=action)


def make_object_deepcopy_compatible(obj: object):
    """This function overwrites the `__deepcopy__` attribute. This is required if e.g.
    the object contains non leaf PyTorch tensors.

    Args:
        obj: An object where a corresponding `__deepcopy__` attributed is added.

    """

    def __deepcopy__(memo):
        detach_all_non_leaf_tensors(obj)
        cls = obj.__class__
        result = cls.__new__(cls)
        memo[id(obj)] = result
        for k, v in obj.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    setattr(obj, "__deepcopy__", __deepcopy__)
    return obj
