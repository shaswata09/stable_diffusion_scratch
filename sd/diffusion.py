import torch
from attention import CrossAttention, SelfAttention
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    """_summary_
    Args:
        nn (nn.Module): The time embedding module is implemented as a module that generates sinusoidal time step embeddings.
    Returns:
        torch.Tensor: The output tensor from the time embedding module, which contains the sinusoidal embeddings for the input time steps.
    """

    def __init__(self, d_embed: int):
        super().__init__()

        # Dimensionality of the time embeddings
        # D_Embed
        # d_embed = 320 is transformed to 4 * d_embed = 1280
        self.linear_1 = nn.Linear(d_embed, 4 * d_embed)
        # Linear layer to project back to 4 * d_embed
        # (4 * D_Embed, 4 * D_Embed)
        self.linear_2 = nn.Linear(4 * d_embed, 4 * d_embed)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # time: (1, 320) -> (1, 1280)

        # (1, 320) -> (1, 1280)
        output = self.linear_1(time)
        # SiLU Activation
        output = F.silu(output)
        # (1, 1280) -> (1, 1280)
        output = self.linear_2(output)
        # Return the time step embeddings
        # (1, 1280)
        return output


class UNET_ResidualBlock(nn.Module):
    """_summary_
    Args:
        nn (nn.Module): The UNET residual block is implemented as a module that processes input feature maps through convolutional layers and incorporates time embeddings.
    Returns:
        torch.Tensor: The output tensor from the UNET residual block, which has the same shape as the input tensor but with enhanced feature representations due to the residual connections and time embeddings.
    """

    def __init__(self, in_channels: int, out_channels: int, n_time: int = 1280):
        super().__init__()

        # Group normalization layer for input features
        self.groupnorm_feature = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        # First convolutional layer
        # (in_channels, out_channels)
        self.conv_feature = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        # Group normalization layer for time embeddings
        # (out_channels)
        self.linear_time = nn.Linear(n_time, out_channels)

        # Second convolutional layer
        # (out_channels, out_channels)
        self.groupnorm_merged = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        # (out_channels, out_channels)
        self.conv_merged = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

        # Residual connection for matching input and output channels
        if in_channels == out_channels:
            # Identity mapping for residual connection when input and output channels are the same
            # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
            self.residual_layer = nn.Identity()
        else:
            # Convolutional layer for residual connection when input and output channels differ
            # (in_channels, out_channels)
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, feature: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # feature: (Batch_Size, in_channels, Height, Width)
        # time: (1, 1280)

        # Store the input for the residual connection
        residue = feature

        # First normalization and convolution
        # (Batch_Size, in_channels, Height, Width) -> (Batch_Size, out_channels, Height, Width)
        feature = self.groupnorm_feature(feature)

        # SiLU Activation
        feature = F.silu(feature)

        # First convolution
        # (Batch_Size, out_channels, Height, Width) -> (Batch_Size, out_channels, Height, Width)
        feature = self.conv_feature(feature)

        # Add time embedding
        time_emb = F.silu(time)
        # (1, out_channels) -> (Batch_Size, out_channels, 1, 1)
        time_emb = self.linear_time(time_emb)

        # Merge time embedding with feature map
        # (Batch_Size, out_channels, Height, Width)
        merged = feature + time_emb.unsqueeze(-1).unsqueeze(-1)

        # Second normalization and convolution
        # (Batch_Size, out_channels, Height, Width) -> (Batch_Size, out_channels, Height, Width)
        merged = self.groupnorm_merged(merged)

        # SiLU Activation
        # (Batch_Size, out_channels, Height, Width) -> (Batch_Size, out_channels, Height, Width)
        merged = F.silu(merged)

        # Second convolution
        # (Batch_Size, out_channels, Height, Width) -> (Batch_Size, out_channels, Height, Width)
        merged = self.conv_merged(merged)

        # Return the output with the residual connection
        # (Batch_Size, out_channels, Height, Width)
        return merged + self.residual_layer(residue)


class UNET_AttentionBlock(nn.Module):
    """_summary_
    Args:
        nn (nn.Module): The UNET attention block is implemented as a module that applies cross-attention to the input feature maps using context embeddings.
    Returns:
        torch.Tensor: The output tensor from the UNET attention block, which has the same shape as the input tensor but with enhanced feature representations due to the attention mechanism.
    """

    def __init__(self, n_heads: int, d_embed: int):
        super().__init__()

        # Group normalization layer for input features
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        channels = n_heads * d_embed

        # Group normalization layer for input features
        self.groupnorm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        # Cross-attention layer
        # (Batch_Size, Seq_Len, D_Embed)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        # Layer normalization before self-attention
        self.layernorm_1 = nn.LayerNorm(channels)
        # Self-attention mechanism
        self.attention_1 = SelfAttention(
            n_heads=n_heads, d_embed=channels, in_proj_bias=False
        )
        # Layer normalization before cross-attention
        self.layernorm_2 = nn.LayerNorm(channels)
        # Cross-attention mechanism
        self.attention_2 = CrossAttention(
            n_heads=n_heads, d_embed=channels, in_proj_bias=False
        )
        # Output projection after attention
        self.layernorm_3 = nn.LayerNorm(channels)
        # Output projection layer
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        # Output projection layer
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        # Final convolutional layer
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, feature: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # feature: (Batch_Size, Features, Height, Width)
        # context: (Batch_Size, Seq_Len, D_Embed)

        # Store the input for the residual connection
        residue_long = feature

        # Initial convolution
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        feature = self.groupnorm(feature)

        # Convolution to prepare for attention
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        feature = self.conv_input(feature)

        n, c, h, w = feature.shape
        # Reshape to (Batch_Size, Height * Width, Features)
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        feature = feature.view(n, c, h * w)

        # Transpose to (Batch_Size, Height * Width, Features)
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        feature = feature.transpose(-1, -2)

        # First Self-Attention Block
        # (Batch_Size, Height * Width, Features)
        residue_short = feature

        # Normalization + Self-Attention with skip connection
        feature = self.layernorm_1(feature)

        # Self-Attention
        feature = self.attention_1(feature)

        # Add the residual connection
        feature = feature + residue_short

        # Cross-Attention Block
        residue_short = feature

        # Normalization + Cross-Attention with skip connection
        feature = self.layernorm_2(feature)

        # Cross-Attention with context
        feature = self.attention_2(feature, context)

        # Add the residual connection
        feature = feature + residue_short

        # Feedforward Block
        residue_short = feature

        # Normalization + Feedforward with skip connection
        feature = self.layernorm_3(feature)

        # Feedforward network with GEGLU activation
        feature, gate = self.linear_geglu_1(feature).chunk(2, dim=-1)

        # GEGLU activation
        feature = feature * F.gelu(gate)

        # Project back to original dimension
        feature = self.linear_geglu_2(feature)

        # Add the residual connection
        feature = feature + residue_short

        # Final projection back to original shape
        feature = feature.transpose(-1, -2)

        # Reshape back to (Batch_Size, Features, Height, Width)
        feature = feature.view(n, c, h, w)

        # Final convolution
        feature = self.conv_output(feature)

        # Return the output with the residual connection
        return feature + residue_long


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (Batch_Size, Features, Height, Width)

        # Upsample the spatial dimensions by a factor of 2 using nearest neighbor interpolation
        # Interpolation is the process of estimating unknown values that fall between known values.
        # In the context of images, upsampling increases the resolution by adding new pixels.
        # Nearest neighbor interpolation is a simple method where each new pixel takes the value of the nearest known pixel.
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    """_summary_
    Args:
        nn (nn.Sequential): The SwitchSequential class is a custom sequential container that can handle different layer types with varying forward method signatures.
    Returns:
        torch.Tensor: The output tensor after passing through all layers in the sequential container.
    """

    # Custom Sequential class to handle different layer types
    def forward(
        self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        for layer in self:
            # Check the type of layer and call with appropriate arguments
            if isinstance(layer, UNET_AttentionBlock):
                # Attention block requires context
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                # Residual block requires time embedding
                x = layer(x, time)
            else:
                # Other layers only require x
                x = layer(x)
        return x


class UNET(nn.Module):
    """_summary_
    Args:
        nn (nn.Module): The UNET model is implemented as a module that processes input embeddings through a series of convolutional and attention layers.
    Returns:
        torch.Tensor: The output tensor from the UNET model, which has the same shape as the input tensor but with enhanced feature representations due to the UNET architecture.
    """

    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList(
            [
                # Initial convolution to increase channel dimension
                # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
                SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
                # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
                SwitchSequential(
                    UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)
                ),
                # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
                SwitchSequential(
                    UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)
                ),
                # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
                SwitchSequential(
                    nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)
                ),
                # (Batch_Size, 320, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
                SwitchSequential(
                    UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)
                ),
                # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
                SwitchSequential(
                    UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)
                ),
                # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
                SwitchSequential(
                    nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)
                ),
                # (Batch_Size, 640, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
                SwitchSequential(
                    UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)
                ),
                # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
                SwitchSequential(
                    UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)
                ),
                # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
                SwitchSequential(
                    nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)
                ),
                # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
                SwitchSequential(UNET_ResidualBlock(1280, 1280)),
                # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
                SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            ]
        )

        self.bottleneck = SwitchSequential(
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280),
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_AttentionBlock(8, 160),
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList(
            [
                # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
                SwitchSequential(UNET_ResidualBlock(2560, 1280)),
                # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
                SwitchSequential(UNET_ResidualBlock(2560, 1280)),
                # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32)
                SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
                # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
                SwitchSequential(
                    UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)
                ),
                # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
                SwitchSequential(
                    UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)
                ),
                # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
                SwitchSequential(
                    UNET_ResidualBlock(1920, 1280),
                    UNET_AttentionBlock(8, 160),
                    Upsample(1280),
                ),
                # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
                SwitchSequential(
                    UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)
                ),
                # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
                SwitchSequential(
                    UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)
                ),
                # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
                SwitchSequential(
                    UNET_ResidualBlock(960, 640),
                    UNET_AttentionBlock(8, 80),
                    Upsample(640),
                ),
                # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
                SwitchSequential(
                    UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)
                ),
                # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
                SwitchSequential(
                    UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)
                ),
                # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
                SwitchSequential(
                    UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)
                ),
            ]
        )

    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 1280)

        skip_connections = []
        # Encoder path
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        for layers in self.encoders:
            # Apply each encoder layer sequentially
            x = layers(x, context, time)
            # Store the output for skip connections
            skip_connections.append(x)

        # Bottleneck
        # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
        x = self.bottleneck(x, context, time)

        # Decoder path
        # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 320, Height / 8, Width / 8)
        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1)
            # Apply each decoder layer sequentially
            x = layers(x, context, time)

        # Return the final output
        # (Batch_Size, 320, Height / 8, Width / 8)
        return x


class UNET_OutputLayer(nn.Module):
    """_summary_
    Args:
        nn (nn.Module): The UNET output layer is implemented as a module that projects the output of the UNET to the desired output shape.
    Returns:
        torch.Tensor: The output tensor from the UNET output layer, which has the specified number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Group normalization layer
        self.groupnorm = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        # Convolutional layer to project to the desired output channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, 320, Height / 8, Width / 8)

        # Apply group normalization
        x = self.groupnorm(x)
        # SiLU Activation
        x = F.silu(x)
        # Project to the desired output channels
        x = self.conv(x)
        # Return the final output
        # (Batch_Size, 4, Height / 8, Width / 8)
        return x


class Diffusion(nn.Module):
    """_summary_
    Args:
        nn (nn.Module): The diffusion model is implemented as a module that consists of multiple transformer layers to process input embeddings.
    Returns:
        torch.Tensor: The output tensor from the diffusion model, which has the same shape as the input tensor but with enhanced feature representations due to the transformer layers.
    """

    def __init__(self):

        # Time embedding module. This module generates time step embeddings for the diffusion process.
        # 320 is the dimensionality of the time embeddings.
        # latent: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        self.time_embedding = TimeEmbedding(320)

        # UNET model for diffusion. This model processes the input embeddings through a series of convolutional and attention layers.
        self.unet = UNET()

        # Final output layer of the UNET model. This layer projects the output of the UNET to the desired output shape.
        self.final = UNET_OutputLayer(320, 4)

    def forward(
        self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        # latent: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 320)

        # Generate time step embeddings
        # (1, 320) -> (1, 1280)
        time_emb = self.time_embedding(time)

        # Process the latent embeddings through the UNET model with time embeddings and context
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        output = self.unet(latent, time_emb, context)

        # Project the output of the UNET to the desired output shape
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        output = self.final(output)

        # Return the final output
        # (Batch_Size, 4, Height / 8, Width / 8)
        return output
