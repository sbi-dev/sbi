import math
from typing import Optional, Tuple

import torch
from torch import nn


class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim: int, power: Optional[float] = 10e4):
        super().__init__()
        self.power = power
        self.embedding_dim = embedding_dim
        div_term = self.power ** (torch.arange(0, embedding_dim, 2) / embedding_dim)
        self.register_buffer("div_term", tensor=div_term, persistent=False)

    def forward(self, x, position_ids: Optional[torch.tensor] = None):
        seq_length = x.shape[-2]
        if position_ids is None:
            position_ids = torch.arange(0, seq_length, 1).to(x)

        div_term = position_ids.view(-1, 1) / self.div_term.to(x)

        pe = torch.zeros_like(x)
        pe[..., 0::2] += torch.cos(div_term)
        pe[..., 1::2] += torch.sin(div_term)

        return x + pe


class RotaryEncoder(nn.Module):
    def __init__(self, embedding_dim: int, power: Optional[float] = 10e4):
        super().__init__()
        self.power = power
        self.embedding_dim = embedding_dim
        div_term = self.power ** (torch.arange(0, embedding_dim, 2) / embedding_dim)
        self.register_buffer("div_term", tensor=div_term, persistent=False)

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, position_ids: Optional[torch.tensor] = None) -> torch.Tensor:
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            x (`torch.Tensor`): query/key tensor
            position_ids (`torch.Tensor`, *optional*):
                specify the position ids
        Returns:
            `(torch.Tensor)` comprising the query/key tensors rotated using the
            Rotary Position Embedding.
        """
        seq_length = x.shape[-2]
        rotary_dim = self.embedding_dim // 2

        if position_ids is None:
            position_ids = torch.arange(0, seq_length, 1).to(x)

        freqs = position_ids.view(-1, 1) / self.div_term.to(x)

        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]

        x_embed = torch.cat(
            [(x_rot * freqs.cos()) + (self.rotate_half(x_rot) * freqs.sin()), x_pass],
            dim=-1,
        )

        return x_embed


POSITION_EMB = {"positional": PositionalEncoder, "rotary": RotaryEncoder}


class FullAttention(nn.Module):
    # Adapted from https://github.com/huggingface/transformers/main/src/transformers/models/phi3/modeling_phi3.py
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config["hidden_size"]
        self.head_dim = config.get(
            "head_dim", config["hidden_size"] // config["num_attention_heads"]
        )
        self.num_heads = config["num_attention_heads"]
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
            config["hidden_size"],
            bias=False,
        )
        self.qkv_proj = nn.Linear(config["hidden_size"], op_size, bias=False)
        self.pos_emb = POSITION_EMB[config.get("pos_emb", "rotary")](
            embedding_dim=self.head_dim, power=config.get("pos_emb_power", 10e4)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
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

        query_states = self.pos_emb(query_states, position_ids)
        key_states = self.pos_emb(key_states, position_ids)

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
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(value_states.dtype)
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
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights
