import torch
from attention import SelfAttention
from torch import nn
from torch.nn import functional as F


class CLIPEmbedding(nn.Module):
    """_summary_
    Args:
        nn (nn.Module): The embedding layer is implemented as a module that maps input token indices to dense vector representations.
    Returns:
        torch.Tensor: The output tensor from the embedding layer, which contains the dense vector representations for the input tokens.
    """

    def __init__(self, vocab_size: int, d_embed: int, max_seq_length: int):
        super().__init__()

        # Token embedding layer
        # (Vocab_Size, D_Embed)
        self.token_embedding = nn.Embedding(vocab_size, d_embed)
        # Positional embedding layer
        # (Max_Seq_Length, D_Embed)
        self.position_embedding = nn.Parameter(torch.zeros((max_seq_length, d_embed)))

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # Get token embeddings
        # tokens: (Batch_Size, Seq_Len) --> (Batch_Size, Seq_Len, D_Embed)
        x = self.token_embedding(tokens)

        # Add positional embeddings
        # (Batch_Size, Seq_Len, D_Embed) + (Seq_Len, D_Embed) --> (Batch_Size, Seq_Len, D_Embed)
        x += self.position_embedding

        return x


class ClipLayer(nn.Module):
    """_summary_
    Args:
        nn (nn.Module): The transformer layer is implemented as a module that consists of a self-attention mechanism followed by a feedforward neural network.
    Returns:
        torch.Tensor: The output tensor from the transformer layer, which has the same shape as the input tensor but with enhanced feature representations due to the attention and feedforward operations.
    """

    def __init__(self, n_heads: int, d_embed: int):
        super().__init__()

        # Layer normalization before self-attention
        # (D_Embed)
        self.layernorm_1 = nn.LayerNorm(d_embed)

        # Self-attention mechanism
        self.self_attention = SelfAttention(n_heads=n_heads, d_embed=d_embed)

        # Layer normalization before feedforward network
        # (D_Embed)
        self.layernorm_2 = nn.LayerNorm(d_embed)
        # Feedforward neural network
        # (D_Embed, 4 * D_Embed) --> (D_Embed)
        self.linear_1 = nn.Linear(d_embed, 4 * d_embed)
        # Linear layer to project back to original embedding dimension
        # (4 * D_Embed, D_Embed)
        self.linear_2 = nn.Linear(4 * d_embed, d_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # (Batch_Size, Seq_Len, D_Embed)

        residue = x

        # Self-attention with layer normalization
        x = self.layernorm_1(x)
        # Self-Attention with causal mask enabled
        x = self.self_attention(x, causal_mask=True)
        # Add the residual connection
        x = x + residue

        # Feedforward Block
        residue = x
        # Layer normalization before feedforward network
        x = self.layernorm_2(x)
        # Feedforward network
        x = self.linear_1(x)

        # x = F.gelu(x)
        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x = x * torch.sigmoid(1.702 * x)  # QuickGELU activation function

        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.linear_2(x)
        # Add the residual connection
        x = x + residue

        # Return the output of the transformer layer
        # (Batch_Size, Seq_Len, D_Embed)
        return x


class CLIP(nn.Module):
    """_summary_
    Args:
        nn (nn.Module): The CLIP model is implemented as a module that processes input images through a series of convolutional and attention layers to produce feature embeddings.
    Returns:
        torch.Tensor: The output tensor from the CLIP model, which contains the feature embeddings for the input images.
    """

    def __init__(self):
        super().__init__()

        # Define the embedding layer
        # (Vocab_Size, D_Embed, Max_Seq_Length)
        self.embedding = CLIPEmbedding(49408, 768, 77)

        # Define the transformer layers
        # 12 layers, each with 768 embedding dimension
        # (Num_Layers, D_Embed)
        self.layers = nn.ModuleList([ClipLayer(12, 768) for _ in range(12)])

        # Define the layer normalization
        # (D_Embed)
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # tokens: (Batch_Size, Seq_Len)
        # Convert tokens to long type
        tokens = tokens.type(torch.long)

        # Get the embeddings for the input tokens
        # (Batch_Size, Seq_Len, D_Embed)
        state = self.embedding(tokens)

        # Process the embeddings through the transformer layers
        # (Batch_Size, Seq_Len, D_Embed)
        for layer in self.layers:
            # Apply each transformer layer to the state
            state = layer(state)

        # Apply layer normalization to the final state
        # (Batch_Size, Seq_Len, D_Embed)
        output = self.layernorm(state)

        # Return the final output
        # (Batch_Size, Seq_Len, D_Embed)
        return output
