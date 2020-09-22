from torch.nn import functional as F

from sbi.utils.regression_net.classifier import ResidualRegressor


def classifier(
    dim
):
    return ResidualRegressor(
        in_features=dim,
        out_features=1,
        hidden_features=200,
        context_features=None,
        num_blocks=2,
        activation=F.relu,
        dropout_probability=0.5,
        use_batch_norm=True,
    )
