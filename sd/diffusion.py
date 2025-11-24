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
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, context, time)

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
