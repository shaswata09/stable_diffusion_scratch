import torch
import torch.nn as nn
from decoder import VAE_AttentionBlock, VAE_ResidualBlock
from torch.nn import functional as F


class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            # (Batch_Size, 3, Height, Width) --> (Batch_Size, 128, Height, Width)
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),
            # (Batch_Size, 128, Height, Width) --> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(in_channels=128, out_channels=128),
            # (Batch_Size, 128, Height, Width) --> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(in_channels=128, out_channels=128),
            # (Batch_Size, 128, Height, Width) --> (Batch_Size, 128, Height/2, Width/2)
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0
            ),
            # (Batch_Size, 128, Height/2, Width/2) --> (Batch_Size, 256, Height/2, Width/2)
            VAE_ResidualBlock(in_channels=128, out_channels=256),
            # (Batch_Size, 256, Height/2, Width/2) --> (Batch_Size, 256, Height/2, Width/2)
            VAE_ResidualBlock(in_channels=256, out_channels=256),
            # (Batch_Size, 256, Height/2, Width/2) --> (Batch_Size, 256, Height/4, Width/4)
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0
            ),
            # (Batch_Size, 256, Height/4, Width/4) --> (Batch_Size, 512, Height/4, Width/4)
            VAE_ResidualBlock(in_channels=256, out_channels=512),
            # (Batch_Size, 512, Height/4, Width/4) --> (Batch_Size, 512, Height/4, Width/4)
            VAE_ResidualBlock(in_channels=512, out_channels=512),
            # (Batch_Size, 512, Height/4, Width/4) --> (Batch_Size, 512, Height/8, Width/8)
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0
            ),
            # (Batch_Size, 512, Height/8, Width/8) --> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(in_channels=512, out_channels=512),
            # (Batch_Size, 512, Height/8, Width/8) --> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(in_channels=512, out_channels=512),
            # (Batch_Size, 512, Height/8, Width/8) --> (Batch_Size, 1024, Height/8, Width/8)
            VAE_ResidualBlock(in_channels=512, out_channels=512),
            # (Batch_Size, 512, Height/8, Width/8) --> (Batch_Size, 512, Height/8, Width/8)
            VAE_AttentionBlock(channels=512),
            # (Batch_Size, 512, Height/8, Width/8) --> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(in_channels=512, out_channels=512),
            # (Batch_Size, 512, Height/8, Width/8) --> (Batch_Size, 512, Height/8, Width/8)
            nn.GroupNorm(num_groups=32, num_channels=512, eps=1e-6),
            # SiLU Activation (It's also known as Swish Activation and is similar to ReLU but smoother.)
            # It helps in better gradient flow and improved performance in deep networks.
            nn.SiLU(),
            # (Batch_Size, 512, Height/8, Width/8) --> (Batch_Size, 8, Height/8, Width/8)
            nn.Conv2d(in_channels=512, out_channels=8, kernel_size=3, padding=1),
            # (Batch_Size, 8, Height/8, Width/8) --> (Batch_Size, 8, Height/8, Width/8)
            # The purpose of this layer is to produce the final output that contains both the mean and log variance for the latent space representation in a VAE.
            # By having out_channels=8, the output tensor can be split into two parts: one for the mean and one for the log variance, each having 4 channels.
            # This is essential for the reparameterization trick used in VAEs, allowing the model to learn a distribution over the latent space.
            # Even though the layer do not change the spatial dimensions, it is crucial for generating the parameters needed for sampling in the VAE framework.
            # Thus, this layer is crucial for enabling the VAE to generate diverse outputs by sampling from the learned latent distribution.
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, 3, Height, Width) --> (Batch_Size, 8, Height/8, Width/8)
        # noise: (Batch_Size, out_channels, Height/8, Width/8)

        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                # If the module has a stride attribute, it indicates a downsampling operation
                # Pad the input tensor to handle cases where height or width is odd. This ensures that the downsampling operation works correctly without losing information.
                # For example, if the input height is 15, downsampling by a factor of 2 would ideally require an even number of pixels. Padding ensures that we can still perform the operation without losing data.
                # The padding is done on the right and bottom edges to maintain the spatial dimensions correctly after downsampling.
                # Pad the right and bottom edges by 1 to handle downsampling of odd dimensions.
                # Padding format: (left, right, top, bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

            # (Batch_Size, 8, Height/8, Width/8) -> Two tensors of shape (Batch_Size, 4, Height/8, Width/8)
            # These represent the mean and log variance for the latent space distribution in a VAE.
            # The mean and log variance are used to parameterize the Gaussian distribution from which latent variables are sampled.
            mean, log_variance = torch.chunk(input=x, chunks=2, dim=1)

            # Clamp log_variance to prevent numerical instability during training. Clamping means restricting the values to a specified range.
            # This ensures that the values of log_variance stay within a reasonable range, avoiding issues like exploding gradients or NaNs.
            # Clamping is done using min and max values. First, log_variance values below -30.0 are set to -30.0, and values above 20.0 are set to 20.0 and rest remain unchanged.
            # Therefore, log_variance tensor values are brought within the range [-30.0, 20.0].

            # (Batch_Size, 4, Height/8, Width/8) --> (Batch_Size, 4, Height/8, Width/8)
            log_variance = torch.clamp(log_variance, min=-30.0, max=20.0)

            # Calculate variance from log_variance
            # (Batch_Size, 4, Height/8, Width/8) --> (Batch_Size, 4, Height/8, Width/8)
            variance = log_variance.exp()

            # Reparameterization trick to sample from the latent space
            # This allows gradients to flow through the sampling process during back propagation.
            # (Batch_Size, 4, Height/8, Width/8) --> (Batch_Size, 4, Height/8, Width/8)
            stdev = variance.sqrt()

            # Z = N(0, 1) --> N(mean, variance)=X
            # X = mean + stdev * Z
            # Generate samples from the latent space using the reparameterization trick
            # This allows gradients to flow through the sampling process during back propagation.
            # (Batch_Size, 4, Height/8, Width/8) --> (Batch_Size, 4, Height/8, Width/8)
            x = mean + stdev * noise

            # Scale the output to maintain numerical stability. Scaling helps in keeping the values within a manageable range, which can improve training stability and convergence.
            # (Batch_Size, 4, Height/8, Width/8) --> (Batch_Size, 4, Height/8, Width/8)
            x *= 0.18215  # Scale factor used in VAE models for numerical stability.

        return x
