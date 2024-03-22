from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pyknos.nflows.nn import nets
from torch import Tensor, nn, relu

from sbi.utils.sbiutils import standardizing_net, z_score_parser
from sbi.utils.user_input_checks import check_data_device, check_embedding_net_device

from sbi.neural_nets.vf_estimators.score_estimator import VEScoreEstimator, VPScoreEstimator, subVPScoreEstimator
from sbi.neural_nets.embedding_nets import GaussianFourierTimeEmbedding

  
class StandardizeInputs(nn.Module):
    def __init__(self, embedding_net_x, embedding_net_y, embedding_net_t, dim_x, dim_y):
        super().__init__()
        self.embedding_net_x = embedding_net_x
        self.embedding_net_y = embedding_net_y
        self.embedding_net_t = embedding_net_t
        self.dim_x = dim_x
        self.dim_y = dim_y

    def forward(self, inputs: list) -> Tensor:
        assert (
            isinstance(inputs, list) and len(inputs) == 3
        ), """Inputs to network must be a list containing raw theta, x, and 1d time."""        
        out = torch.cat(
            [
                self.embedding_net_x(inputs[0]),
                self.embedding_net_y(inputs[1]),
                self.embedding_net_t(inputs[2]),
            ],
            dim=1,
        )
        return out

def build_input_layer(
    batch_x: Tensor,
    batch_y: Tensor,
    t_embedding_dim: int,
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    embedding_net_x: nn.Module = nn.Identity(),
    embedding_net_y: nn.Module = nn.Identity(),
) -> nn.Module:
    """Builds input layer for vector field regression, including time embedding, and optionally z-scores.

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        t_embedding_dim: Dimensionality of the time embedding.
        z_score_x: Whether to z-score xs passing into the network, can be one of:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_y: Whether to z-score ys passing into the network, same options as
            z_score_x.        
        embedding_net_x: Optional embedding network for x.
        embedding_net_y: Optional embedding network for y.

    Returns:
        Input layer that concatenates x, y, and time embedding, optionally z-scores.
    """
    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    if z_score_x_bool:
        embedding_net_x = nn.Sequential(
            standardizing_net(batch_x, structured_x), embedding_net_x
        )

    z_score_y_bool, structured_y = z_score_parser(z_score_y)
    if z_score_y_bool:
        embedding_net_y = nn.Sequential(
            standardizing_net(batch_y, structured_y), embedding_net_y
        )
    embedding_net_t = GaussianFourierTimeEmbedding(t_embedding_dim)
    input_layer = StandardizeInputs(
        embedding_net_x, embedding_net_y, embedding_net_t, dim_x=batch_x.shape[1], dim_y=batch_y.shape[1]
    )
    return input_layer

# def build_mlp_regression(
#     batch_x: Tensor,
#     batch_y: Tensor,
#     z_score_x: Optional[str] = "independent",
#     z_score_y: Optional[str] = "independent",
#     t_embedding_dim: int = 16,
#     hidden_features: int = 50,
#     embedding_net_x: nn.Module = nn.Identity(),
#     embedding_net_y: nn.Module = nn.Identity(),
# ) -> nn.Module:
#     """Builds MLP vector regression network.

#     Args:
#         batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
#         batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
#         z_score_x: Whether to z-score xs passing into the network, can be one of:
#             - `none`, or None: do not z-score.
#             - `independent`: z-score each dimension independently.
#             - `structured`: treat dimensions as related, therefore compute mean and std
#             over the entire batch, instead of per-dimension. Should be used when each
#             sample is, for example, a time series or an image.
#         z_score_y: Whether to z-score ys passing into the network, same options as
#             z_score_x.
#         t_embedding_dim: Dimensionality of the time embedding.
#         hidden_features: Number of hidden features.
#         embedding_net_x: Optional embedding network for x.
#         embedding_net_y: Optional embedding network for y.

#     Returns:
#         Neural network.
#     """
#     check_data_device(batch_x, batch_y)
#     check_embedding_net_device(embedding_net=embedding_net_x, datum=batch_y)
#     check_embedding_net_device(embedding_net=embedding_net_y, datum=batch_y)

#     # Infer the output dimensionalities of the embedding_net by making a forward pass.
#     x_dim = batch_x.shape[1]
#     x_numel = embedding_net_x(batch_x[:1]).numel()
#     y_numel = embedding_net_y(batch_y[:1]).numel()    

#     neural_net = nn.Sequential(
#         nn.Linear(x_numel + y_numel + t_embedding_dim, hidden_features),
#         nn.BatchNorm1d(hidden_features),
#         nn.ReLU(),
#         nn.Linear(hidden_features, hidden_features),
#         nn.BatchNorm1d(hidden_features),
#         nn.ReLU(),
#         nn.Linear(hidden_features, x_dim),
#     )

#     input_layer = build_input_layer(
#         batch_x, batch_y, t_embedding_dim, z_score_x, z_score_y, embedding_net_x, embedding_net_y
#     )

#     neural_net = nn.Sequential(input_layer, neural_net)
#     return neural_net


def build_score_estimator(
    batch_x: Tensor,
    batch_y: Tensor,
    sde_type: Optional[str] = 'vp',
    score_net: Optional[Union[str, nn.Module]] = 'mlp',
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    t_embedding_dim: int = 16,
    num_layers: int = 3,
    hidden_features: int = 50,
    embedding_net_x: nn.Module = nn.Identity(),
    embedding_net_y: nn.Module = nn.Identity(),
    **kwargs,
):
    """Builds score estimator for score-based generative models."""
    check_data_device(batch_x, batch_y)
    check_embedding_net_device(embedding_net=embedding_net_x, datum=batch_y)
    check_embedding_net_device(embedding_net=embedding_net_y, datum=batch_y)

    input_layer = build_input_layer(
        batch_x, batch_y, t_embedding_dim, z_score_x, z_score_y, embedding_net_x, embedding_net_y
    )

    # Infer the output dimensionalities of the embedding_net by making a forward pass.
    x_dim = batch_x.shape[1]
    x_numel = embedding_net_x(batch_x[:1]).numel()
    y_numel = embedding_net_y(batch_y[:1]).numel()
    if score_net == 'mlp':       
       score_net = MLP(x_numel + y_numel + t_embedding_dim, x_dim, hidden_dim=hidden_features, num_layers=num_layers)
    elif score_net == 'resnet':       
       raise NotImplementedError
    elif isinstance(score_net, nn.Module):
        pass
    else:
        raise ValueError(f"Invalid score network: {score_net}")
    
    if sde_type == 'vp':
        estimator = VPScoreEstimator
    elif sde_type == 've':
        estimator = VEScoreEstimator
    elif sde_type == 'subvp':
        estimator = subVPScoreEstimator
    else:
        raise ValueError(f"SDE type: {sde_type} not supported.")

    neural_net = nn.Sequential(input_layer, score_net)
    return estimator(neural_net, batch_y.shape, **kwargs)


class MLP(nn.Module):
  """Simple fully connected neural network."""
  def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=3):
    super().__init__()
    self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
    for _ in range(num_layers - 1):
      self.layers.append(nn.Linear(hidden_dim, hidden_dim))
    self.layers.append(nn.Linear(hidden_dim, output_dim))
  
  def forward(self, x):
    for layer in self.layers[:-1]:
      x = F.relu(layer(x))
    return self.layers[-1](x)
