from os import remove
from numpy import isin
import torch
from torch import nn
from torch.distributions import TransformedDistribution, Distribution
from torch.distributions.transforms import (
    Transform,
    ComposeTransform,
    IndependentTransform,
)

from copy import deepcopy

from typing import Optional, Iterable, Callable, Dict


def docstring_parameter(*sub, base_doc: str = None) -> Callable:
    """This is an decorater which can be used to use variables within a docstring,
    similar to string formating.



    Args:
        base_doc: You can provide a base_docstring, which is used instead of the own.


    """

    def dec(obj):
        if base_doc is None:
            obj.__doc__ = obj.__doc__.format(*sub)
        else:
            obj.__doc__ = base_doc.format(*sub)
        return obj

    return dec


def filter_kwrags_for_func(f: Callable, kwargs: Dict) -> Dict:
    """This function will filter a dictionary of possible arguments for arguments the
    function can use.



    Args:
        f: Function for which kwargs are filtered
        kwargs: Possible kwargs for function

    Returns:
        dict: Subset of kwargs, which the function f can take as aruments.

    """
    args = f.__code__.co_varnames
    new_kwargs = dict([(key, val) for key, val in kwargs.items() if key in args])
    return new_kwargs


def get_parameters(t: Transform) -> Iterable:
    """Recursive helper function which can be used to extract parameters from
    TransformedDistributions.



    Args:
        t: A TorchTransform object, which is scanned for the "parameters" attribute.

    Yields:
        Iterator[Iterable]: Generator of parameters.
    """
    if hasattr(t, "parameters"):
        yield from t.parameters()
    elif isinstance(t, ComposeTransform):
        for part in t.parts:
            yield from get_parameters(part)
    elif isinstance(t, IndependentTransform):
        yield from get_parameters(t.base_transform)
    else:
        pass


def get_modules(t: Transform) -> Iterable:
    """Recursive helper function which can be used to extract modules from TransformedDistributions.



    Args:
        t: A TorchTransform object, which is scanned for the "modules" attribute.

    Yields:
        Iterator[Iterable]: Generator of TransformModules
    """
    if isinstance(t, nn.Module):
        yield t
    elif isinstance(t, ComposeTransform):
        for part in t.parts:
            yield from get_modules(part)
    elif isinstance(t, IndependentTransform):
        yield from get_modules(t.base_transform)
    else:
        pass


def check_parameters_modules_attribute(q: Distribution) -> None:
    """Checks a parameterized distribution object for valid 'parameters' and 'modules'.

    Args:
        q: Distribution object
    """

    if not hasattr(q, "parameters"):
        raise ValueError(
            "The variational distribution requires an parameters attribute, which returns an iterable of parameters"
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
        ), "Nothing to train, atleast one of the parameters must have an enable gradient!"
    if not hasattr(q, "modules"):
        raise ValueError(
            "The variational distribution requires an modules attribute, which returns an iterable of parameters"
        )
    else:
        assert isinstance(q.modules, Callable), "The parameters must be callable"
        modules = q.modules()
        assert isinstance(
            modules, Iterable
        ), "The parameters return value must be iterable"
        for m in modules:
            assert isinstance(
                m, torch.nn.Module
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

    samples = q.sample((2,))
    samples_prior = prior.sample((2,))
    assert all(
        prior.support.check(samples)
    ), "The support of q must match that of the prior"
    assert (
        samples.shape == samples_prior.shape
    ), "Something is wrong with sample shape and event_shape or batch_shape attributes."


def check_variational_distribution(q: Distribution, prior: Distribution) -> None:
    """Runs all basic checks such the q is 'valid'.

    Args:
        q: Variational distribution which is checked
        prior: Prior to check certain attributes which should be satisfied.

    """
    check_parameters_modules_attribute(q)
    check_sample_shape_and_support(q, prior)


def add_parameters_module_attributes(
    q: Distribution, parameters: Callable, modules: Callable
):
    """Sets the attribute 'parameters' and 'modules' on q

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
    """A function that will add 'parameters' and 'modules' to q automatically, if q is a TransformedDistribution.

    Args:
        q: Variational distribution instances TransformedDistribution.


    """

    def parameters():
        if hasattr(q.base_dist, "parameters"):
            yield from q.base_dist.parameters()
        for t in q.transforms:
            yield from get_parameters(t)

    def modules():
        if hasattr(q.base_dist, "modules"):
            yield from q.base_dist.modules()
        for t in q.transforms:
            yield from get_modules(t)

    add_parameters_module_attributes(q, parameters, modules)


def adapt_and_check_variational_distributions(
    q: Distribution, q_kwargs: dict, prior: Distribution, link_transform: Callable
) -> TransformedDistribution:
    """This will adapt a distribution to be compatible with DivergenceOptimizers.
    Especially it will make sure that the distribution has parameters and that it
    satisfies obvious contraints which a posterior must satisfy i.e. the support must be
    equal to that of the prior.

    Args:
        q: Variational distribution.
        q_kwargs: Keyword arguments.
        prior: Prior distribution
        theta_transform: Theta transformation.

    Returns:
        TransformedDistribution: Compatible variational distribution.

    """
    if isinstance(q, TransformedDistribution):
        if q.support != prior.support:
            q = TransformedDistribution(q.base_dist, q.transforms + [link_transform])
        add_parameter_attributes_to_transformed_distribution(q)

    else:
        if "parameters" in q_kwargs:
            params = q_kwargs["parameters"]
        else:
            params = []

        def parameters():
            return params

        if "modules" in q_kwargs:
            mod = q_kwargs["modules"]
        else:
            mod = []

        def modules():
            return mod

        # Compatible with deepcopy, tensors must use clone instead of deepcopy...
        def __deepcopy__(self, *args, **kwargs):
            cls = self.__class__
            result = cls.__new__(cls)
            if hasattr(self, "__dict__"):
                for k, v in self.__dict__:
                    if isinstance(v, torch.Tensor):
                        setattr(result, k, v.clone())
                    else:
                        setattr(result, k, deepcopy(v))
            else:
                return deepcopy(self)
            return result

        q.__deepcopy__ = __deepcopy__

        q = TransformedDistribution(q, [link_transform])
        add_parameters_module_attributes(q, parameters, modules)

    check_variational_distribution(q, prior)

    return q


def make_sure_nothing_in_cache(q: TransformedDistribution) -> None:
    """This function tries to ensure that no non-leaf tensors are within the
    Distribution object. This is required to ensure that 'deepcopy' protocols work.



    Args:
        q: Distribution


    """
    q.clear_cache()
    # The original methods can miss some parts..
    for t in q.transforms:
        # Compose transforms are not cleared correctly using q.clear_cache...
        if isinstance(t, torch.distributions.transforms.IndependentTransform):
            t = t.base_transform
        if isinstance(t, torch.distributions.transforms.ComposeTransform):
            for t_i in t.parts:
                t_i._cached_x_y = None, None

        t._cached_x_y = None, None

        t_dict = t.__dict__
        for key in t_dict:
            if "cache" in key or "det" in key:
                obj = t_dict[key]
                if torch.is_tensor(obj):
                    t_dict[key] = torch.zeros_like(obj)
    # Overwrite with non gradient thinks (THIS MAY BE SUFFICIENT TODO!)
    q.sample()
