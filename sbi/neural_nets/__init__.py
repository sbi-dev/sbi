from sbi.neural_nets.factory import (
    classifier_nn,
    flowmatching_nn,
    likelihood_nn,
    posterior_nn,
    posterior_score_nn,
)

class CNNEmbedding():
    """Exists only to raise an explicit error for imports of embedding networks."""
    def __init__(self, *args, **kwargs):
        raise ValueError(
            "As of sbi v0.23.0, you have to import embedding networks from "
            "`sbi.neural_nets.embedding_nets`. For example, use: "
            "`from sbi.neural_nets.embedding_nets import CNNEmbedding`"
        )
    

class FCEmbedding():
    """Exists only to raise an explicit error for imports of embedding networks."""
    def __init__(self, *args, **kwargs):
        raise ValueError(
            "As of sbi v0.23.0, you have to import embedding networks from "
            "`sbi.neural_nets.embedding_nets`. For example, use: "
            "`from sbi.neural_nets.embedding_nets import FCEmbedding`"
        )
    

class PermutationInvariantEmbedding():
    """Exists only to raise an explicit error for imports of embedding networks."""
    def __init__(self, *args, **kwargs):
        raise ValueError(
            "As of sbi v0.23.0, you have to import embedding networks from "
            "`sbi.neural_nets.embedding_nets`. For example, use: "
            "`from sbi.neural_nets.embedding_nets import PermutationInvariantEmbedding`"
        )
