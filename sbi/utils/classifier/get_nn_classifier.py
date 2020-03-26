from torch.nn import functional as F
from sbi.utils.classifier.classifier import ResidualClassifier


def classifier(
    dim
):
    return ResidualClassifier(
        in_features=dim,
        out_features=2,
        hidden_features=1000,
        context_features=None,
        num_blocks=5,
        activation=F.relu,
        dropout_probability=0.5,
        use_batch_norm=True,
    )
