from sbi.neural_nets.factory import (
    classifier_nn,
    flowmatching_nn,
    likelihood_nn,
    posterior_nn,
    posterior_score_nn,
)


def __getattr__(name):
    if name in ["CNNEmbedding", "FCEmbedding", "PermutationInvariantEmbedding"]:
        raise ImportError(
            "As of sbi v0.23.0, you have to import embedding networks from "
            "`sbi.neural_nets.embedding_nets`. For example, use: "
            f"`from sbi.neural_nets.embedding_nets import {name}`"
        )
    elif name == "classifier_nn":
        return classifier_nn
    elif name == "flowmatching_nn":
        return flowmatching_nn
    elif name == "likelihood_nn":
        return likelihood_nn
    elif name == "posterior_nn":
        return posterior_nn
    elif name == "posterior_score_nn":
        return posterior_score_nn
    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")
