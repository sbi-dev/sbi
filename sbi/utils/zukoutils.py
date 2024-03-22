from torch import Tensor
from zuko.flows import LazyTransform, Unconditional
from zuko.transforms import Transform


# This is a temporary wrapper for the Unconditional class in zuko.
# Avoids pyright error of zuko Flows requiring a LazyTransform as input.
class UnconditionalLazyTransform(LazyTransform):
    def __init__(self, unconditional: Unconditional):
        super().__init__()
        self.unconditional = unconditional

    def forward(self, c: Tensor) -> Transform:
        return self.unconditional.meta(
            *self.unconditional._parameters.values(),
            *self.unconditional._buffers.values(),
            **self.unconditional.kwargs,
        )
