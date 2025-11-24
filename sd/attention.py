import math

import torch
from torch import nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    """_summary_
    Args:
        nn (nn.Module): The self-attention mechanism is implemented as a module that computes attention weights and applies them to the input embeddings.
    Returns:
        torch.Tensor: The output tensor from the self-attention mechanism, which has the same shape as the input tensor but with enhanced feature representations due to the attention mechanism.
    """

    def __init__(
        self,
        n_heads: int,
        d_embed: int,
        in_proj_bias: bool = True,
        out_proj_bias: bool = True,
    ):
        super().__init__()
        # Input projection layer to compute queries, keys, and values from the input embeddings.
        # (Batch_Size, Seq_Len, D_Embed) -> (Batch_Size, Seq_Len, 3*D_Embed)
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        # Output projection layer to project the output of the attention mechanism back to the original embedding dimension.
        # (Batch_Size, Seq_Len, D_Embed) -> (Batch_Size, Seq_Len, D_Embed)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        # Number of attention heads and dimension per head
        # (N_Heads, D_Head)
        self.n_heads = n_heads
        # Dimension of each attention head
        # D_Head
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        # x: (Batch_Size, Seq_Len, D_Embed)

        # Store the input shape for later use
        input_shape = x.shape

        # Extract batch size, sequence length, and embedding dimension from the input tensor
        # (Batch_Size, Seq_Len, D_Embed)
        batch_size, seq_len, d_embed = input_shape

        # Compute queries, keys, and values from the input tensor
        # (Batch_Size, Seq_Len, 3*D_Embed)
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        # Reshape queries, keys, and values to separate heads
        # (Batch_Size, Seq_Len, D_Embed) -> (Batch_Size, Seq_Len, D_Embed * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, D_Embed)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # Reshape and transpose
        # (Batch_Size, N_Heads, D_Embed) --> (Batch_Size, N_Heads, Seq_Len, N_Heads, D_Embed/N_Heads) --> (Batch_Size, N_Heads, Seq_Len, D_Embed/N_Heads)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # Scaled dot-product attention
        # Scale queries to prevent large dot-product values
        # (Batch_Size, N_Heads, Seq_Len, D_Embed/N_Heads)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu_(1)
            # Apply the mask to the weight tensor, setting masked positions to negative infinity
            weight = weight.masked_fill(mask, -torch.inf)

        # normalize by sqrt of d_head
        weight /= math.sqrt(self.d_head)

        # Apply softmax to obtain attention weights
        weight = F.softmax(weight, dim=-1)

        # Compute the output by applying attention weights to the values
        # (Batch_Size, N_Heads, Seq_Len, Seq_Len) @ (Batch_Size, N_Heads, Seq_Len, D_Embed/N_Heads) --> (Batch_Size, N_Heads, Seq_Len, D_Embed/N_Heads)
        output = weight @ v

        # (Batch_Size, N_Heads, Seq_Len, D_Embed/N_Heads) --> (Batch_Size, Seq_Len, N_Heads, D_Embed/N_Heads)
        output = output.transpose(1, 2)

        # (Batch_Size, Seq_Len, N_Heads, D_Embed/N_Heads) --> (Batch_Size, Seq_Len, D_Embed)
        output = output.reshape(input_shape)

        # Project the output back to the original embedding dimension
        # (Batch_Size, Seq_Len, D_Embed) --> (Batch_Size, Seq_Len, D_Embed)
        output = self.out_proj(output)

        # Return the output tensor
        # (Batch_Size, Seq_Len, D_Embed)
        return output
