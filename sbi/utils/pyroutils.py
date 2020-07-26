from typing import Callable, Optional, Union, Dict, Any, Tuple, Union, cast, List, Sequence, TypeVar

from pyro import distributions as dist
from pyro import poutine as poutine
from torch.distributions import biject_to


def get_transforms(model: Callable, *model_args: Any, **model_kwargs: Any):
    """Get automatic transforms to unbounded space

    Args:
        model: Pyro model
        model_args: Arguments passed to model
        model_args: Keyword arguments passed to model
    
    Example:
        ```python
        def prior():
            return pyro.sample("theta", pyro.distributions.Uniform(0., 1.))
            
        transform_to_unbounded = get_transforms(prior)["theta"]
        ```
    """
    transforms = {}

    model_trace = poutine.trace(model).get_trace(*model_args, **model_kwargs)

    for name, node in model_trace.iter_stochastic_nodes():
        fn = node["fn"]
        transforms[name] = biject_to(fn.support).inv

    return transforms
