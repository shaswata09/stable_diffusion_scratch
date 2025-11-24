import torch
from attention import SelfAttention
from torch import nn
from torch.nn import functional as F


class VAE_AttentionBlock(nn.Module):
    """_summary_

    Args:
        nn (nn.Module): The attention block is implemented as a module that applies group normalization followed by self-attention.
        It takes an input tensor of shape (Batch_Size, Channels, Height, Width) and processes it to capture long-range dependencies in the feature maps.
    Returns:
        torch.Tensor: The output tensor from the attention block, which has the same shape as the input tensor, but with enhanced feature representations due to the attention mechanism.
    """

    def __init__(self, channels: int):
        super().__init__()
        # Group Normalization layer to normalize the input across groups of channels, which helps stabilize training and improve convergence.
        # (Batch_Size, Channels, Height, Width) -> (Batch_Size, Channels, Height, Width)
        self.groupnorm = nn.GroupNorm(num_groups=32, num_channels=channels)
        # Self-Attention mechanism to capture long-range dependencies in the feature maps.
        # (Batch_Size, Channels, Height, Width) -> (Batch_Size, Channels, Height, Width)
        self.attention = SelfAttention(n_heads=1, d_embed=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channels, Height, Width)

        residue = x  # Save the input for the residual connection

        # n, c, h, w represent the batch size, number of channels, height, and width of the input tensor respectively.
        n, c, h, w = x.shape

        # (Batch_Size, Channels, Height, Width) -> (Batch_Size, Channels, Height*Width)
        x = x.view(n, c, h * w)

        # Transpose to (Batch_Size, Height*Width, Channels)
        # (Batch_Size, Channels, Height*Width) -> (Batch_Size, Height*Width, Channels)
        x = x.transpose(-1, -2)

        # (Batch_Size, Height*Width, Channels) -> (Batch_Size, Height*Width, Channels)
        x = self.attention(x)

        # Transpose back to (Batch_Size, Channels, Height*Width)
        # (Batch_Size, Height*Width, Channels) -> (Batch_Size, Channels, Height*Width)
        x = x.transpose(-1, -2)

        # (Batch_Size, Channels, Height*Width) -> (Batch_Size, Channels, Height, Width)
        x = x.view(n, c, h, w)

        x += residue  # Add the residue (input) to the output of the attention mechanism

        return x  # Return the output with the residual connection applied


class VAE_ResidualBlock(nn.Module):
    """_summary_

    Args:
        nn (nn.Module): The residual block is implemented as a module that applies two convolutional layers with group normalization and SiLU activation.
        It takes an input tensor of shape (Batch_Size, In_Channels, Height, Width) and processes it to produce an output tensor of shape (Batch_Size, Out_Channels, Height, Width).
    Returns:
        torch.Tensor: The output tensor from the residual block, which has the same spatial dimensions as the input tensor but may have a different number of channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # First convolutional layer
        # The group normalization layer normalizes the input across groups of channels, which helps stabilize training and improve convergence.
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        self.groupnorm_1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        # Convolutional layer to transform input channels to output channels
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        self.conv_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )

        # Second convolutional layer
        # The second group normalization layer further normalizes the output from the first convolutional layer.
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        self.groupnorm_2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        # Convolutional layer to refine the features
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        self.conv_2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        # Residual connection layer
        # This layer ensures that the input can be added to the output even if the number of channels changes.
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        if in_channels == out_channels:
            # If the number of input and output channels are the same, use an identity mapping.
            self.residual_layer = nn.Identity()
        else:
            # If the number of input and output channels are different, use a 1x1 convolution to match dimensions.
            self.residual_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channels, Height, Width)

        # Get the residue for the residual connection
        # This ensures that the input can be added to the output even if the number of channels changes.
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        residue = self.residual_layer(x)

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        x = self.groupnorm_1(x)
        # Apply SiLU activation function
        # # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        x = F.silu(x)

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.conv_1(x)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.groupnorm_2(x)
        # Apply SiLU activation function
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = F.silu(x)
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.conv_2(x)

        # Add the residue (input) to the output of the convolutional layers
        # This is the core idea of a residual block, helping to mitigate the vanishing gradient problem and allowing for deeper networks.
        # (Batch_Size, Out_Channels, Height, Width)
        return x + residue  # Residual connection
