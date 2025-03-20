import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
}


class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim: int, power: Optional[float] = 10e4):
        super().__init__()
        self.power = power
        self.embedding_dim = embedding_dim
        div_term = self.power ** (torch.arange(0, embedding_dim, 2) / embedding_dim)
        self.register_buffer("div_term", tensor=div_term, persistent=False)

    def forward(self, x, position_ids: Optional[torch.Tensor] = None):
        seq_length = x.shape[-2]
        if position_ids is None:
            position_ids = torch.arange(0, seq_length, 1).to(x)

        div_term = position_ids.view(-1, 1) / self.div_term.to(x)

        pe = torch.zeros_like(x)
        pe[..., 0::2] += torch.cos(div_term)
        pe[..., 1::2] += torch.sin(div_term)

        return x + pe


class DummyEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, **kwargs):
        return x


class RotaryEncoder(nn.Module):
    def __init__(self, embedding_dim: int, power: Optional[float] = 10e4):
        super().__init__()
        self.power = power
        self.embedding_dim = embedding_dim
        div_term = self.power ** (
            torch.arange(0, embedding_dim, 2) / embedding_dim
        ).repeat(2)
        self.register_buffer("div_term", tensor=div_term, persistent=False)

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
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

        if position_ids is None:
            position_ids = torch.arange(0, seq_length, 1).to(x)

        freqs = position_ids.view(-1, 1) / self.div_term.to(x)

        x_embed = x * freqs.cos() + self.rotate_half(x) * freqs.sin()

        return x_embed


POSITION_EMB = {
    "positional": PositionalEncoder,
    "rotary": RotaryEncoder,
    "none": DummyEncoder,
}


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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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


class MLP(nn.Module):
    # Adapted from https://github.com/huggingface/transformers/main/src/transformers/models/phi3/modeling_phi3.py
    """
    Feed-forward layer which can be replace by a custom implementation
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.gate_up_proj = nn.Linear(
            config["hidden_size"], 2 * config["intermediate_size"], bias=False
        )
        self.down_proj = nn.Linear(
            config["intermediate_size"], config["hidden_size"], bias=False
        )

        self.activation_fn = ACT2FN[config["mlp_activation"]]

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        up_states = self.gate_up_proj(x)
        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)

        return self.down_proj(up_states)


# Copied from https://github.com/huggingface/transformers/main/src/transformers/models/phi3/modeling_phi3.py
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class MoeBlock(nn.Module):
    # Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens).
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config["hidden_size"]
        self.ffn_dim = config["intermediate_size"]
        self.num_experts = config["num_local_experts"]
        self.top_k = config["num_experts_per_tok"]

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList([MLP(config) for _ in range(self.num_experts)])

        # Jitter parameters
        self.jitter_noise = config.get("router_jitter_noise", 0.0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
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
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states


class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config["hidden_size"]

        self.self_attn = FullAttention(config=config, layer_idx=layer_idx)

        self.ffn = FFN[config.get("ffn", "mlp")](config)
        self.input_layernorm = RMSNorm(
            config["hidden_size"], eps=config.get("rms_norm_eps", 1e-05)
        )
        self.post_attention_layernorm = RMSNorm(
            config["hidden_size"], eps=config.get("rms_norm_eps", 1e-05)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch,
            seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
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

        return outputs


FFN = {
    "mlp": MLP,
    "moe": MoeBlock,
}


class TransformerEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx)
            for layer_idx in range(config["num_hidden_layers"])
        ])
        self.is_causal = config["is_causal"]

        self.norm = RMSNorm(
            config["hidden_size"], eps=config.get("rms_norm_eps", 1e-05)
        )

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
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = (
                causal_mask.clone()
            )  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = (
                causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            )
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[
                :, :, :, :mask_length
            ].masked_fill(padding_mask, min_dtype)
        return causal_mask

    def forward(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: Optional[torch.tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch,
            seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)`
            output_attentions (`bool`, *optional*):
                Whether or not to return the attention tensors
            kwargs (`dict`, *optional*):
                Arbitrary kwargs
        """
        if self.is_causal:
            dtype, device = input_embeddings.dtype, input_embeddings.device
            attention_mask = self.causal_mask(
                sequence_length=input_embeddings.shape[1],
                dtype=input_embeddings.dtype,
                device=device,
                batch_size=input_embeddings.shape[0],
                min_dtype=torch.finfo(dtype).min,
            )
        else:
            attention_mask = None

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        hidden_states = input_embeddings
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if output_attentions or output_hidden_states:
            return hidden_states[:, -1, :], (all_hidden_states, all_self_attns)

        return hidden_states[:, -1, :]
