from typing import Optional

from pyknos.nflows import flows, transforms
from torch import Tensor, nn

import sbi.utils as utils
from sbi.neural_nets.density_estimators.mdn_net_flex import MultivariateGaussianMDNFlex
from sbi.utils.user_input_checks import check_data_device, check_embedding_net_device


def build_mdn_flex(
    batch_x: Tensor,
    batch_y: Tensor,
    partition: Tensor,
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    hidden_features: int = 50,
    num_components: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    **kwargs,
) -> nn.Module:
    """Builds MDN_Flex p(x|y).

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        partition: Tensor of ints. partition of the x space.
            indicating the partition of the parameter (feature) space
        z_score_x: Whether to z-score xs passing into the network, can be one of:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_y: Whether to z-score ys passing into the network, same options as
            z_score_x.
        hidden_features: Number of hidden features.
        num_components: Number of components.

        embedding_net: Optional embedding network for y.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for MDNs and are therefore ignored.

    Returns:
        Neural network.
    """

    x_numel = batch_x[0].numel()
    # Infer the output dimensionality of the embedding_net by making a forward pass.
    check_data_device(batch_x, batch_y)
    check_embedding_net_device(embedding_net=embedding_net, datum=batch_y)
    y_numel = embedding_net(batch_y[:1]).numel()

    transform = transforms.IdentityTransform()

    # transform theta
    # TODO: check if this can deal with the nan inputs!
    z_score_x_bool, structured_x = utils.z_score_parser(z_score_x)
    if z_score_x_bool:
        transform_zx = utils.standardizing_transform(batch_x, structured_x)
        transform = transforms.CompositeTransform([transform_zx, transform])

    # transform "data"
    z_score_y_bool, structured_y = utils.z_score_parser(z_score_y)
    if z_score_y_bool:
        embedding_net = nn.Sequential(
            utils.standardizing_net(batch_y, structured_y), embedding_net
        )

    distribution = MultivariateGaussianMDNFlex(
        features=x_numel,
        partition=partition,
        context_features=y_numel,
        hidden_features=hidden_features,
        hidden_net=nn.Sequential(
            nn.Linear(y_numel, hidden_features),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
        ),
        num_components=num_components,
        custom_initialization=True,
    )

    neural_net = flows.Flow(transform, distribution, embedding_net)

    return neural_net
