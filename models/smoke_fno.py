import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2D Fourier layer. It does FFT, linear transform, and Inverse FFT."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeff's of real signals (rfft2)
        x_ft = torch.fft.rfft2(x)

        # Number of actual modes to keep (at most self.modes)
        m1 = min(self.modes1, x_ft.size(-2))
        m2 = min(self.modes2, x_ft.size(-1))

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x_ft.size(-2),
            x_ft.size(-1),
            dtype=torch.cfloat,
            device=x.device,
        )

        # Extract and multiply top-left and bottom-left Fourier modes
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNOBlock(nn.Module):
    """Core FNO block with Spectral Conv, Skip connection and Activation."""

    def __init__(self, channels, modes1, modes2):
        super(FNOBlock, self).__init__()
        self.spec_conv = SpectralConv2d(channels, channels, modes1, modes2)
        self.skip_conv = nn.Conv2d(channels, channels, 1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        return F.gelu(self.norm(self.spec_conv(x) + self.skip_conv(x)))


class SmokeWorldModel(nn.Module):
    """Drone Smoke World Model using FNO and U-Net skip connections.
    Identified in prompt as "FNO-Primary" with specific input/output requirements.
    """

    def __init__(self, in_channels=8, out_channels=3, latent_dim=128, modes=12):
        super(SmokeWorldModel, self).__init__()

        # 1. Lifting: Project 8 channels to latent space
        self.lifting = nn.Sequential(
            nn.Conv2d(in_channels, latent_dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(latent_dim // 2, latent_dim, 1),
        )

        # 2. UNet-style Encoder (Spatial Refinement Prep)
        # Using strides to create hierarchy for the skip connections
        self.enc1 = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1, stride=2),  # 64 -> 32
            nn.BatchNorm2d(latent_dim),
            nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1, stride=2),  # 32 -> 16
            nn.BatchNorm2d(latent_dim),
            nn.GELU(),
        )

        # 3. FNO Core (Spectral Blocks) - Operating at the lowest spatial resolution (bottleneck)
        # Or at full resolution? Prompt says "Primary motor... modela advección global".
        # If I put it at 16x16, the modes should be close to the resolution.
        self.fno_blocks = nn.Sequential(
            FNOBlock(latent_dim, modes, modes),
            FNOBlock(latent_dim, modes, modes),
            FNOBlock(latent_dim, modes, modes),
            FNOBlock(latent_dim, modes, modes),
        )

        # 4. Decoder with Spatial Skip-Connections
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dim, latent_dim, kernel_size=4, stride=2, padding=1
            ),  # 16 -> 32
            nn.BatchNorm2d(latent_dim),
            nn.GELU(),
        )
        # Skip connection from enc1 joins here (latent_dim + latent_dim -> latent_dim)
        self.dec2_proj = nn.Conv2d(latent_dim * 2, latent_dim, 1)

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dim, latent_dim, kernel_size=4, stride=2, padding=1
            ),  # 32 -> 64
            nn.BatchNorm2d(latent_dim),
            nn.GELU(),
        )
        # Skip connection from lifting joins here
        self.dec1_proj = nn.Conv2d(latent_dim * 2, latent_dim, 1)

        # 5. Output Projection: mu_t, sigma_t, mu_{t+k}
        # Result channels: out_channels
        self.projection = nn.Sequential(
            nn.Conv2d(latent_dim, 64, 1), nn.GELU(), nn.Conv2d(64, out_channels, 1)
        )

    def forward(self, x):
        """Input: (B, 8, 64, 64)
        Returns: (B, 3, 64, 64) -> [mu_t, log_sigma_t, mu_t_k]
        """
        # Lifting
        lat = self.lifting(x)  # (B, 128, 64, 64)

        # Encoder
        s1 = self.enc1(lat)  # (B, 128, 32, 32)
        s2 = self.enc2(s1)  # (B, 128, 16, 16)

        # FNO Core (Bottleneck)
        bottleneck = self.fno_blocks(s2)  # (B, 128, 16, 16)

        # Decoder
        up2 = self.dec2(bottleneck)  # (B, 128, 32, 32)
        up2 = self.dec2_proj(torch.cat([up2, s1], dim=1))

        up1 = self.dec1(up2)  # (B, 128, 64, 64)
        up1 = self.dec1_proj(torch.cat([up1, lat], dim=1))

        # Projection
        out = self.projection(up1)

        # Split into mu, log_sigma, mu_next
        # We can further process this in the training loop (e.g., softplus for sigma)
        return out


if __name__ == "__main__":
    batch_size = 4
    model = SmokeWorldModel(in_channels=8, latent_dim=128, modes=12)
    dummy_input = torch.randn(batch_size, 8, 64, 64)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 3, 64, 64)
    print("Model Architecture OK")
