from zuko.flows import LazyTransform, Unconditional


# This is a temporary wrapper for the Unconditional class in zuko.
# Avoids pyright error of zuko Flows requiring a LazyTransform as input.
class UnconditionalLazyTransform(Unconditional, LazyTransform):
    pass
