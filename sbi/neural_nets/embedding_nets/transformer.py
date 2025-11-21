# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class PositionalEncoder(nn.Module):
    def __init__(self, head_dim: int, base: Optional[float] = 10e4):
        """
        Position encoding as described by Vaswani et. al.
        https://arxiv.org/abs/1706.03762
        Args:
            head_dim (int): dimensionality of the key/query vectors
            base (float, *optional*): base used to create the positional encodings
        """
        super().__init__()

        self.base = base
        self.head_dim = head_dim
        if self.head_dim % 2 != 0:
            raise ValueError(f"`head_dim`:{self.head_dim} must be even")

        div_term = self.base ** (torch.arange(0, head_dim, 2) / head_dim)
        self.register_buffer("div_term", tensor=div_term, persistent=False)

    def forward(
        self, x: torch.FloatTensor, position_ids: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x (torch.FloatTensor): query/key  of shape `(bsz, num_heads, seq_len,
            head_dim)`
            position_ids (torch.tensor, *optional*): specify the position ids, by
            default constructs 0-sequence_length
        Returns:
            `(torch.Tensor)` query/key tensors with standard additive positional
            encoding
        """
        seq_length = x.shape[-2]
        if position_ids is None:
            position_ids = torch.arange(0, seq_length, 1).to(x)

        div_term = position_ids.view(-1, 1) / self.div_term.to(x)

        pe = torch.zeros_like(x)
        pe[..., 0::2] += torch.cos(div_term)
        pe[..., 1::2] += torch.sin(div_term)

        return x + pe


class IdentityEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: torch.FloatTensor, **kwargs):
        """
        No transformation of the input is applied.
        Args:
            x `(torch.FloatTensor)`
        Return
            `(torch.FloatTensor)`
        """
        return x


class RotaryEncoder(nn.Module):
    def __init__(self, head_dim: int, base: Optional[float] = 10e4):
        """
        Rotary position encoding as described by Su et. al.
        https://arxiv.org/abs/2104.09864
        Args:
            head_dim (int): feature dimension of the key/query vector
            base (float): base to be used to create the positional encodings
        """
        super().__init__()

        self.base = base
        self.head_dim = head_dim
        if self.head_dim % 2 != 0:
            raise ValueError(f"`head_dim`:{self.head_dim} must be even")
        div_term = self.base ** (torch.arange(0, head_dim, 2) / head_dim).repeat(2)
        self.register_buffer("div_term", tensor=div_term, persistent=False)

    def rotate_half(self, x: torch.FloatTensor):
        """
        Rotates half the hidden dims of the input.
        Args:
            x (torch.FloatTensor): query/key tensors of shape `(bsz, num_heads,
            seq_len, head_dim)`
        Returns
            `(torch.Tensor)` query/key rotated tensors
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self, x: torch.FloatTensor, position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Applies Rotary Position Encoding to the query and key tensors.

        Args:
            x (`torch.Tensor`): query/key tensor of shape `(bsz, num_heads, seq_len,
            head_dim)`
            position_ids (`torch.Tensor`, *optional*):
                specify the position ids, by default constructs 0-seq_len
        Returns:
            `(torch.Tensor)` comprising the query/key tensors rotated using the
            Rotary Position Encoding.
        """

        seq_length = x.shape[-2]

        if position_ids is None:
            position_ids = torch.arange(0, seq_length, 1).to(x)

        freqs = position_ids.view(-1, 1) / self.div_term.to(x)

        x_embed = x * freqs.cos() + self.rotate_half(x) * freqs.sin()

        return x_embed


class FullAttention(nn.Module):
    # Adapted from https://github.com/huggingface/transformers/main/src/transformers/
    # models/phi3/modeling_phi3.py

    def __init__(self, config):
        """Multi-headed attention from 'Attention Is All You Need' paper"""

        super().__init__()
        self.config = config
        self.feature_space_dim = config["feature_space_dim"]
        head_dim = config["head_dim"]
        if head_dim is None:
            if (config["feature_space_dim"] % config["num_attention_heads"]) != 0:
                raise ValueError(
                    "If not providing head_dim, ensure `feature_space_dim` is "
                    "divisible by `num_attention_heads`"
                )
            head_dim = config["feature_space_dim"] // config["num_attention_heads"]

        self.head_dim = head_dim
        self.num_heads = config["num_attention_heads"]
        if (config["num_attention_heads"] % config["num_key_value_heads"]) != 0:
            raise ValueError(
                "`num_attention_heads` must be divisible by `num_key_value_heads`"
            )
        self.num_key_value_groups = (
            config["num_attention_heads"] // config["num_key_value_heads"]
        )
        self.num_key_value_heads = config["num_key_value_heads"]
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config["attention_dropout"]

        op_size = config["num_attention_heads"] * self.head_dim + 2 * (
            config["num_key_value_heads"] * self.head_dim
        )
        self.o_proj = nn.Linear(
            config["num_attention_heads"] * self.head_dim,
            config["feature_space_dim"],
            bias=False,
        )
        # This single layer performs the query, key and value projections
        # The output is then spit into key_states, query_states, and value_states
        # with the corresponding dimensions
        self.qkv_proj = nn.Linear(config["feature_space_dim"], op_size, bias=False)
        pos_emb = {
            "positional": PositionalEncoder,
            "rotary": RotaryEncoder,
            "none": IdentityEncoder,
        }
        self.pos_emb = pos_emb[config["pos_emb"]](
            head_dim=self.head_dim, base=config["pos_emb_base"]
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Computes the attention
        Args:
            hidden_states (`torch.FloatTensor`): query/key tensor of shape `(bsz,
            seq_len, feature_space_dim)`
            position_ids (`torch.Tensor`, *optional*): specify the position ids, by
            default constructs 0-sequence_length
            attention_mask (`torch.Tensor`) : Attention mask of shape `(batch_size,
            sequence_length, feature_space_dim)`
            output_attentions (bool) : return the attention weights, cannot be used
            within the NPE/NRE/NLE pipelines,
            use it for analyzing the embedding modules
        Returns:
            `(torch.Tensor, torch.Tensor)` or `(torch.Tensor)` attention output and
            optionally the attention weights

        """

        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim

        query_states = qkv[..., :query_pos]
        key_states = qkv[
            ..., query_pos : query_pos + self.num_key_value_heads * self.head_dim
        ]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        query_states = self.pos_emb(query_states, position_ids=position_ids)
        key_states = self.pos_emb(key_states, position_ids=position_ids)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = key_states.repeat_interleave(
            dim=1, repeats=self.num_key_value_groups
        )
        value_states = value_states.repeat_interleave(
            dim=1, repeats=self.num_key_value_groups
        )

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[..., : key_states.shape[-2]]
            attn_weights += causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(
            value_states.dtype
        )
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                "`attn_output` should be of size "
                + f"{(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                + f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.feature_space_dim)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class MLP(nn.Module):
    # Adapted from https://github.com/huggingface/transformers/main/src/
    # transformers/models/phi3/modeling_phi3.py

    def __init__(self, config):
        """
        Feed-forward layer which can be replaced by a custom implementation
        f(x):
            R^{feature_space_dim} -> R^{intermediate_size}
            R^{intermediate_size} -> R^{feature_space_dim}
        """
        super().__init__()

        self.config = config
        self.gate_up_proj = nn.Linear(
            config["feature_space_dim"], 2 * config["intermediate_size"], bias=False
        )
        self.down_proj = nn.Linear(
            config["intermediate_size"], config["feature_space_dim"], bias=False
        )
        if config["mlp_activation"] == "gelu":
            self.activation_fn = F.gelu
        elif config["mlp_activation"] == "relu":
            self.activation_fn = F.relu
        else:
            raise ValueError(
                "Unsupported activation function, currently supported: `gelu, relu`"
            )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            hidden_states (torch.FloatTensor): output from the attention layer of
            shape `(batch_size, sequence_length, feature_space_dim)`
        Returns:
            `(torch.FloatTensor)`
        """
        up_states = self.gate_up_proj(
            hidden_states
        )  # projection of hidden_states to 2*intermediate_size
        gate, up_states = up_states.chunk(
            2, dim=-1
        )  # split the resulting vector in two (intermediate_size,intermediate_size)
        up_states = up_states * self.activation_fn(
            gate
        )  # use one of the splits as input to the activation function and the other
        # to scale it

        return self.down_proj(up_states)


# Copied from https://github.com/huggingface/transformers/blob/main/src/
# transformers/models/phi3/modeling_phi3.py
class RMSNorm(nn.Module):
    def __init__(self, feature_space_dim, eps: float):
        """
        RMSNorm is equivalent to T5LayerNorm
        Variant of layer normalization https://arxiv.org/abs/1607.06450
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(feature_space_dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states (torch.FloatTensor): input of shape `(batch_size,
            sequence_length, feature_space_dim)`
            RMS normalization with per dimension scaling (self.weight)
        Returns:
            `(torch.FloatTensor)`
        """

        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class MoeBlock(nn.Module):
    # Adapted from https://github.com/huggingface/transformers/blob/main/src/
    # transformers/models/mixtral/modeling_mixtral.py
    # https://arxiv.org/abs/2401.04088
    def __init__(self, config):
        super().__init__()
        """
        Mixture of experts implementation with full capacity (no dropped tokens).
        `num_local_experts` : specifies the total number of experts available
        `num_experts_per_tok`: number of experts each token is assigned to
        `router_jitter_noise` : noise to be added at training time before routing
        the tokens to experts
        """
        self.hidden_dim = config["feature_space_dim"]
        self.ffn_dim = config["intermediate_size"]
        self.num_experts = config["num_local_experts"]
        self.top_k = config[
            "num_experts_per_tok"
        ]  # Each token is assigned independently to experts
        if self.top_k > self.num_experts:
            raise ValueError(
                "Each token cannot be assigned to more that num_local_experts"
            )

        # gating function to determine the experts to be used for each token
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # experts can be replaced by a custom implementation of feed forward layer
        self.experts = nn.ModuleList([MLP(config) for _ in range(self.num_experts)])

        # Jitter parameters
        self.jitter_noise = config["router_jitter_noise"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            `hidden_states` : input of shape `(batch_size, sequence_length,
            hidden_dim)`
        Returns:
            `final_hidden_states` : output of shape `(batch_size, sequence_length,
            hidden_dim)`
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation
        # on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            if top_x.numel() > 0:
                # Only proceed if there are tokens assigned to this expert
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = (
                    expert_layer(current_state) * routing_weights[top_x, idx, None]
                )

                # However `index_add_` only support torch tensors for indexing so
                # we'll use the `top_x` tensor here.
                final_hidden_states.index_add_(
                    0, top_x, current_hidden_states.to(hidden_states.dtype)
                )
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        """
        Transformer block containing an attention and feed-forward layer
        It also contains 2 learnable layer-norms
        """
        self.feature_space_dim = config["feature_space_dim"]

        self.self_attn = FullAttention(config=config)
        ffn = {
            "mlp": MLP,
            "moe": MoeBlock,
        }
        self.ffn = ffn[config["ffn"]](config)
        self.input_layernorm = RMSNorm(
            config["feature_space_dim"], eps=config["rms_norm_eps"]
        )
        self.post_attention_layernorm = RMSNorm(
            config["feature_space_dim"], eps=config["rms_norm_eps"]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states (`torch.Tensor`): input to the layer of shape `(batch,
            seq_len, embed_dim)`
            attention_mask (`torch.Tensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)`
            output_attentions (`bool`, *optional*):
                Whether or not to return the attention tensors
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        else:
            outputs += (None,)

        return outputs


class ViTEmbeddings(nn.Module):
    # Adapted from https://github.com/huggingface/transformers/blob/main/src/
    # transformers/models/vit/modeling_vit.py
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels,
    height, width)` into the initial `hidden_states` (patch embeddings)
    of shape `(batch_size, seq_length, feature_space_dim)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        """
        image_size (int) is the expected size of the square image
        patch_size (int) is the expected size of the square patch
        """
        image_size, patch_size = config["image_size"], config["patch_size"]
        num_channels, feature_space_dim = (
            config["num_channels"],
            config["feature_space_dim"],
        )
        num_patches = (image_size // patch_size) ** 2

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1, config["feature_space_dim"])
        )
        self.projection = nn.Conv2d(
            num_channels, feature_space_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["feature_space_dim"]))
        self.dropout = nn.Dropout(config["vit_dropout"])

    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings,
        to be able to use the model on higher resolution images.
        Args:
            embeddings (torch.FloatTensor): embedding patches generated after
            applying the CNN filters on the input image
            height (int): height of the input image
            width (int): width of the input image
        """
        # Adapted from:
        # https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac95
        # 2ab558447af1fa1365362a/vision_transformer.py

        # https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fb
        # adf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py

        num_positions = self.position_embeddings.shape[1] - 1

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(
            1, sqrt_num_positions, sqrt_num_positions, dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        Images of shape `(batch_size, num_channels, height, width)`
        Return input_embeddings of shape `(batch_size, seq_length, feature_space_dim)`
        to be consumed by a Transformer.
        """
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values "
                "match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )

        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        embeddings = embeddings + self.interpolate_pos_encoding(
            embeddings, height, width
        )

        embeddings = self.dropout(embeddings)

        return embeddings


class TransformerEmbedding(nn.Module):
    r"""
    Transformer-based embedding network for **time series** and **image** data.

    This module provides a flexible embedding architecture that supports both
    (1) 1D / multivariate time series (e.g., experimental trials, temporal signals),
    and
    (2) image inputs via a lightweight Vision Transformer (ViT)-style patch embedding.

    It is designed for simulation-based inference (SBI) workflows where raw
    observations must be encoded into fixed-dimensional embeddings before passing
    them to a neural density estimator.

    Parameters
    ----------
    pos_emb :
        Positional embedding type. One of ``{"rotary", "positional", "none"}``.
    pos_emb_base :
        Base frequency for rotary positional embeddings.
    rms_norm_eps :
        Epsilon for RMSNorm layers.
    router_jitter_noise :
        Noise added when routing tokens to MoE experts.
    vit_dropout :
        Dropout applied inside ViT patch embedding layers.
    mlp_activation :
        Activation used inside the feedforward blocks.
    is_causal :
        If ``True``, applies a causal mask during attention (useful for time-series).
    vit :
        If ``True``, enables Vision Transformer mode for 2D image inputs.
    num_hidden_layers :
        Number of transformer encoder blocks.
    num_attention_heads :
        Number of self-attention heads.
    num_key_value_heads :
        Number of KV heads (for multi-query attention).
    intermediate_size :
        Hidden dimension of feedforward network (or MoE experts).
    ffn :
        Feedforward type. One of ``{"mlp", "moe"}``.
    head_dim :
        Per-head embedding dimension. If ``None``, inferred as
        ``feature_space_dim // num_attention_heads``.
    attention_dropout :
        Dropout used inside the attention mechanism.
    feature_space_dim :
        Dimensionality of the token embeddings flowing through the transformer.
        - For time-series, this is the model dimension.
        - For images (``vit=True``), this is the post-patch-projection embedding size.
    final_emb_dimension :
        Output embedding dimension. Defaults to ``feature_space_dim // 2``.
    image_size :
        Input image height/width (only if ``vit=True``).
    patch_size :
        ViT patch size (only if ``vit=True``).
    num_channels :
        Number of image channels for ViT mode.
    num_local_experts :
        Number of MoE experts (only relevant when ``ffn="moe"``).
    num_experts_per_tok :
        How many experts each token is routed to in MoE mode.

    Notes
    -----
    **Time-series mode (``vit=False``)**
    - Inputs of shape ``(batch, seq_len)`` (scalar series) are automatically
      projected to ``(batch, seq_len, feature_space_dim)``.
    - Inputs of shape ``(batch, seq_len, features)`` are used as-is.
    - Causal masking is applied if ``is_causal=True`` (default).
    - Suitable for experimental trials, temporal dynamics, or sets of sequential
      observations.

    **Image mode (``vit=True``)**
    - Inputs must have shape ``(batch, channels, height, width)``.
    - Images are patchified, linearly projected, and fed to the transformer.
    - Causal masking is disabled in this mode.

    **Output**
    The embedding is obtained by selecting the final token and applying a linear
    projection, resulting in a tensor of shape:

    ``(batch, final_emb_dimension)``

    Example
    -------
    **1D time-series (default mode)**::

        from sbi.neural_nets.embedding_nets import TransformerEmbedding
        import torch

        x = torch.randn(16, 100)       # (batch, seq_len)
        emb = TransformerEmbedding(feature_space_dim=64)
        z = emb(x)

    **Image input (ViT-style)**::

        from sbi.neural_nets.embedding_nets import TransformerEmbedding
        import torch

        x = torch.randn(8, 3, 64, 64)  # (batch, C, H, W)
        emb = TransformerEmbedding(
            vit=True,
            image_size=64,
            patch_size=8,
            num_channels=3,
            feature_space_dim=128,
        )
        z = emb(x)
    """

    def __init__(
        self,
        *,
        pos_emb: str = "rotary",
        pos_emb_base: float = 10e4,
        rms_norm_eps: float = 1e-05,
        router_jitter_noise: float = 0.0,
        vit_dropout: float = 0.5,
        mlp_activation: str = "gelu",
        is_causal: bool = True,
        vit: bool = False,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 12,
        intermediate_size: int = 256,
        ffn: str = "mlp",
        head_dim: Optional[int] = None,
        attention_dropout: float = 0.5,
        feature_space_dim: int,
        final_emb_dimension: Optional[int] = None,
        image_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        num_channels: Optional[int] = None,
        num_local_experts: Optional[int] = None,
        num_experts_per_tok: Optional[int] = None,
    ):
        super().__init__()
        """
        Main class for constructing a transformer embedding

        Args:
            pos_emb: position encoding to be used, currently available:
                {"rotary", "positional", "none"}
            pos_emb_base: base used to construct the positinal encoding
            rms_norm_eps: noise added to the rms variance computation
            ffn: feedforward layer after used after computing the attention:
                {"mlp", "moe"}
            mlp_activation: activation function to be used within the ffn
                layer
            is_causal: specifies whether causal mask should be created
            vit: specifies the whether a convolutional layer should be used for
                processing images, inspired by the vision transformer
            num_hidden_layers: number of transformer blocks
            num_attention_heads: number of attention heads
            num_key_value_heads: number of key/value heads
            feature_space_dim: dimension of the feature vectors
            intermediate_size: hidden size of the feedforward layer
                head_dim: dimension key/query vectors
            attention_dropout: value for the dropout of the attention layer

        MoE:
            router_jitter_noise: noise added before routing the input vectors
                to the experts
            num_local_experts: total number of experts
            num_experts_per_tok: number of experts each token is assigned to

        ViT
            feature_space_dim: dimension of the feature vectors after
                preprocessing the images
            image_size: dimension of the squared image used to created
                the positional encoders
                a rectagular image can be used at training/inference time by
                resampling the encoders
            patch_size: size of the square patches used to create the
                positional encoders
            num_channels: number of channels of the input image
            vit_dropout: value for the dropout of the attention layer
        """

        self.config = dict(
            pos_emb=pos_emb,
            pos_emb_base=pos_emb_base,
            rms_norm_eps=rms_norm_eps,
            router_jitter_noise=router_jitter_noise,
            vit_dropout=vit_dropout,
            mlp_activation=mlp_activation,
            is_causal=is_causal,
            vit=vit,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            ffn=ffn,
            head_dim=head_dim,
            attention_dropout=attention_dropout,
            feature_space_dim=feature_space_dim,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            num_local_experts=num_local_experts,
            num_experts_per_tok=num_experts_per_tok,
        )

        self.preprocess = ViTEmbeddings(self.config) if vit else IdentityEncoder()

        self._supports_scalar_series = not vit
        if self._supports_scalar_series:
            self.scalar_projection = nn.Linear(
                1, feature_space_dim
            )  # proj 1D → model dim

        self.layers = nn.ModuleList([
            TransformerBlock(self.config) for _ in range(num_hidden_layers)
        ])
        self.is_causal = is_causal and not vit

        self.norm = RMSNorm(feature_space_dim, eps=rms_norm_eps)

        if final_emb_dimension is None:
            final_emb_dimension = feature_space_dim // 2

        if not vit and final_emb_dimension > feature_space_dim:
            raise ValueError(
                "The final embedding dimension should be "
                "equal or smaller than the input dimension"
            )
        self.aggregator = nn.Linear(
            feature_space_dim,
            final_emb_dimension,
        )
        self.causal_mask_cache_ = (None, None, None)

    def causal_mask(
        self,
        sequence_length: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        min_dtype: float,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        causal_mask = torch.full(
            (sequence_length, sequence_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )  # square causal mask filed with min_dtype
        if sequence_length != 1:
            causal_mask = torch.triu(
                causal_mask, diagonal=1
            )  # zero out lower triangular matrix
        causal_mask = causal_mask[None, None, :, :].expand(
            batch_size, 1, -1, -1
        )  # causal mask
        if attention_mask is not None:
            causal_mask = (
                causal_mask.clone()
            )  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = (
                causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            )  # use attention mask to mask padding tokens
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[
                :, :, :, :mask_length
            ].masked_fill(padding_mask, min_dtype)
        return causal_mask

    def forward(
        self,
        input: torch.Tensor,
        attention_mask: Optional[torch.tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_attention_mask: Optional[bool] = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            input:
                input of shape `(batch, seq_len, feature_space_dim)`
                or `(batch, num_channels, height, width)` if using ViT
            attention_mask:
                attention mask of size `(batch_size, sequence_length)`
            output_attentions:
                Whether or not to return the attention tensors
            cache_attention_mask:
                Whether or not to cache the expanded attention mask, useful if using
                multiple batched with identical input shapes
            kwargs:
                Arbitrary kwargs
        """

        input = self.preprocess(input)

        if self._supports_scalar_series and input.ndim == 2:
            input = input.unsqueeze(-1)  # (B, T, 1)
            input = self.scalar_projection(input)  # (B, T, feature_space_dim)

        if self.is_causal:
            dtype, device = input.dtype, input.device

            cached_attn_mask, cached_mask, cached_shape = self.causal_mask_cache_
            if (
                cache_attention_mask
                and input.shape == cached_shape
                and cached_attn_mask == attention_mask
            ):
                attention_mask = cached_mask
            else:
                cache = (
                    (attention_mask.clone(),) if attention_mask is not None else (None,)
                )
                attention_mask = self.causal_mask(
                    sequence_length=input.shape[1],
                    attention_mask=attention_mask,
                    dtype=input.dtype,
                    device=device,
                    batch_size=input.shape[0],
                    min_dtype=torch.finfo(dtype).min,
                )
                cache += (attention_mask,)
                cache += (input.shape,)
                if cache_attention_mask:
                    self.causal_mask_cache_ = cache
        else:
            attention_mask = None

        # decoder layers
        hidden_states = input
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        return self.aggregator(hidden_states[:, -1, :])
