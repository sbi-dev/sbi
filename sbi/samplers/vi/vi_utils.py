import torch
from torch import nn
from torch.distributions import TransformedDistribution, Distribution
from torch.distributions.transforms import Transform, ComposeTransform

from typing import Optional, Iterable, Callable, Dict


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
    """Recursive helper function to determine all possible parameters in a Transformed
    Distribution object"""
    if hasattr(t, "parameters"):
        yield from t.parameters()
    elif isinstance(t, ComposeTransform):
        for part in t.parts:
            yield from get_parameters(part)
    else:
        pass


def get_modules(t: Transform) -> Iterable:
    """Recursive helper function to determine all modules"""
    if isinstance(t, nn.Module):
        yield t
    elif isinstance(t, ComposeTransform):
        for part in t.parts:
            yield from get_modules(part)
    else:
        pass


def check_parameters_modules_attribute(q: Distribution):
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


def check_sample_shape_and_support(q: Distribution, prior: Distribution):
    assert (
        q.event_shape == prior.event_shape
    ), "The event shape of q must match that of the prior"
    assert (
        q.batch_shape == prior.batch_shape
    ), "The batch sahpe of q must match that of the prior"

    samples = q.sample((10000,))
    assert all(
        prior.support.check(samples)
    ), "The support of q must match that of the prior"


def check_variational_distribution(q: Distribution, prior: Distribution):
    check_parameters_modules_attribute(q)
    check_sample_shape_and_support(q, prior)


def add_parameters_module_attributes(
    q: Distribution, parameters: Callable, modules: Callable
):
    setattr(q, "parameters", parameters)
    setattr(q, "modules", modules)


def add_parameter_attributes_to_transformed_distribution(q: TransformedDistribution):
    def parameters():
        for t in q.transforms:
            yield from get_parameters(t)

    def modules():
        for t in q.transforms:
            yield from get_modules(t)

    add_parameters_module_attributes(q, parameters, modules)


def adapt_and_check_variational_distributions(
    q: Distribution, q_kwargs: dict, prior: Distribution, theta_transform: Callable
):
    if isinstance(q, TransformedDistribution):
        if q.support != prior.support:
            q = TransformedDistribution(q.base_dist, q.transforms + [theta_transform])
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

        # Compatible with deepcopy
        def __deepcopy__(*args, **kwargs):
            for key, vals in q.__dict__.items():
                if isinstance(vals, torch.Tensor):
                    q.__dict__[key] = vals.clone()
            return q

        q.__deepcopy__ = __deepcopy__

        q = TransformedDistribution(q, [theta_transform])
        add_parameters_module_attributes(q, parameters, modules)

    # check_variational_distribution(q, prior)

    return q


def make_sure_nothing_in_cache(q):
    """This may be used before a 'deepcopy' call, as non leaf tensors (which are in the
    cache) do not support the deepcopy protocol...
    Unfortunaltly the q.clear_cache() function does only remove a subset of cached tensors."""
    q.clear_cache()
    # The original methods can miss some parts..
    for t in q.transforms:
        t._cached_x_y = None, None
        # Compose transforms are not cleared correctly using q.clear_cache...
        if isinstance(t, torch.distributions.transforms.IndependentTransform):
            t = t.base_transform
        if isinstance(t, torch.distributions.transforms.ComposeTransform):
            for t_i in t.parts:
                t_i._cached_x_y = None, None

        t_dict = t.__dict__
        for key in t_dict:
            if "cache" in key or "det" in key:
                obj = t_dict[key]
                if torch.is_tensor(obj):
                    t_dict[key] = torch.zeros_like(obj)
