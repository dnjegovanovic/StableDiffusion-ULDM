import torch
import torch.nn as nn

from ddpm_model.models.UNetBlocks import DownSamplingBlock, BottleNeck, UpSamplingBlock


class VariationalAutoEncoder(nn.Module):
    """Variational Autoencoder (VAE) with UNet-like architecture components.

    Args:
        im_channels (int): Number of channels in the input image
        model_config (dict): Configuration dictionary containing:
            - encoder_channels (list): Channel sizes for encoder downsampling blocks
            - bottleneck_channels (list): Channel sizes for bottleneck blocks
            - downsampling_steps (list): Whether to downsample at each encoder block
            - num_down_layers (int): Number of layers per downsampling block
            - num_bottleneck_layers (int): Number of layers per bottleneck block
            - num_up_layers (int): Number of layers per upsampling block
            - use_attention (list): Whether to use attention in each encoder block
            - latent_channels (int): Dimensionality of latent space
            - norm_groups (int): Number of groups for GroupNorm
            - num_heads (int): Number of attention heads
    """

    def __init__(self, im_channels, model_config):
        super().__init__()
        self.im_channels = im_channels
        # Encoder configuration
        self.encoder_channels = model_config["encoder_channels"]
        self.bottleneck_channels = model_config["bottleneck_channels"]
        self.downsampling_steps = model_config["downsampling_steps"]
        self.use_attention = model_config["use_attention"]

        # Layer configuration
        self.num_down_layers = model_config["num_down_layers"]
        self.num_bottleneck_layers = model_config["num_bottleneck_layers"]
        self.num_up_layers = model_config["num_up_layers"]

        # Latent space configuration
        self.latent_channels = model_config["latent_channels"]
        self.norm_groups = model_config["norm_groups"]
        self.num_heads = model_config["num_heads"]

        # Validate configuration
        assert len(self.downsampling_steps) == len(self.encoder_channels) - 1
        assert len(self.use_attention) == len(self.encoder_channels) - 1
        assert self.bottleneck_channels[0] == self.encoder_channels[-1]
        assert self.bottleneck_channels[-1] == self.encoder_channels[-1]

        # Create encoder and decoder components
        self._create_encoder()
        self._create_decoder()

    def _create_encoder(self):
        """Build encoder components: downsampling blocks and bottleneck."""
        # Initial convolution
        self.encoder_input_conv = nn.Conv2d(
            in_channels=self.im_channels,
            out_channels=self.encoder_channels[0],
            kernel_size=3,
            padding=1,
        )

        # Downsampling blocks
        self.encoder_blocks = nn.ModuleList()
        for in_ch, out_ch, downsample, attn in zip(
            self.encoder_channels[:-1],
            self.encoder_channels[1:],
            self.downsampling_steps,
            self.use_attention,
        ):
            self.encoder_blocks.append(
                DownSamplingBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    down_sample=downsample,
                    num_layers=self.num_down_layers,
                    num_heads=self.num_heads,
                    use_attn=attn,
                    grp_norm_chanels=self.norm_groups,
                )
            )

        # Bottleneck blocks
        self.bottleneck_blocks = nn.ModuleList()
        for in_ch, out_ch in zip(
            self.bottleneck_channels[:-1], self.bottleneck_channels[1:]
        ):
            self.bottleneck_blocks.append(
                BottleNeck(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    num_layers=self.num_bottleneck_layers,
                    num_heads=self.num_heads,
                    grp_norm_chanels=self.norm_groups,
                )
            )

        # Latent space projection
        self.encoder_norm = nn.GroupNorm(self.norm_groups, self.encoder_channels[-1])
        self.encoder_output_conv = nn.Conv2d(
            in_channels=self.encoder_channels[-1],
            out_channels=2 * self.latent_channels,
            kernel_size=3,
            padding=1,
        )
        self.latent_projection = nn.Conv2d(
            2 * self.latent_channels, 2 * self.latent_channels, kernel_size=1
        )

    def _create_decoder(self):
        """Build decoder components: projection, bottleneck, and upsampling blocks."""
        # Latent projection
        self.latent_in_projection = nn.Conv2d(
            self.latent_channels, self.latent_channels, kernel_size=1
        )

        # Decoder input processing
        self.decoder_input_conv = nn.Conv2d(
            in_channels=self.latent_channels,
            out_channels=self.bottleneck_channels[-1],
            kernel_size=3,
            padding=1,
        )

        # Reverse bottleneck blocks
        self.decoder_bottlenecks = nn.ModuleList()
        for in_ch, out_ch in zip(
            reversed(self.bottleneck_channels[1:]),
            reversed(self.bottleneck_channels[:-1]),
        ):
            self.decoder_bottlenecks.append(
                BottleNeck(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    num_layers=self.num_bottleneck_layers,
                    num_heads=self.num_heads,
                    grp_norm_chanels=self.norm_groups,
                )
            )

        # Upsampling blocks
        self.decoder_blocks = nn.ModuleList()
        for in_ch, out_ch, upsample, attn in zip(
            reversed(self.encoder_channels[1:]),
            reversed(self.encoder_channels[:-1]),
            reversed(self.downsampling_steps),
            reversed(self.use_attention),
        ):
            self.decoder_blocks.append(
                UpSamplingBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    up_sample=upsample,
                    num_layers=self.num_up_layers,
                    num_heads=self.num_heads,
                    use_attn=attn,
                    grp_norm_chanels=self.norm_groups,
                )
            )

        # Output processing
        self.decoder_norm = nn.GroupNorm(self.norm_groups, self.encoder_channels[0])
        self.decoder_output_conv = nn.Conv2d(
            in_channels=self.encoder_channels[0],
            out_channels=self.im_channels,
            kernel_size=3,
            padding=1,
        )

    def encode(self, x):
        """Encode input into latent parameters.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            tuple: (latent_sample, latent_params) where:
                latent_sample (Tensor): Sampled latent vectors
                latent_params (Tensor): Concatenated mean and logvar
        """
        # Feature extraction
        x = self.encoder_input_conv(x)
        for block in self.encoder_blocks:
            x = block(x)

        # Bottleneck processing
        for bottleneck in self.bottleneck_blocks:
            x = bottleneck(x)

        # Latent parameter estimation
        x = self.encoder_norm(x)
        x = nn.SiLU()(x)
        latent_params = self.encoder_output_conv(x)
        latent_params = self.latent_projection(latent_params)

        # Reparameterization trick
        mean, logvar = torch.chunk(latent_params, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent_sample = mean + eps * std

        return latent_sample, latent_params

    def decode(self, z):
        """Decode latent sample into reconstructed input.

        Args:
            z (Tensor): Latent sample tensor

        Returns:
            Tensor: Reconstructed output
        """
        # Latent processing
        z = self.latent_in_projection(z)
        z = self.decoder_input_conv(z)

        # Bottleneck processing
        for bottleneck in self.decoder_bottlenecks:
            z = bottleneck(z)

        # Upsampling
        for block in self.decoder_blocks:
            z = block(z)

        # Output processing
        z = self.decoder_norm(z)
        z = nn.SiLU()(z)
        return self.decoder_output_conv(z)

    def forward(self, x):
        """VAE forward pass.

        Args:
            x (Tensor): Input tensor

        Returns:
            tuple: (reconstruction, latent_params)
        """
        latent_sample, latent_params = self.encode(x)
        print(f"Latent shape:{latent_sample.shape}")
        reconstruction = self.decode(latent_sample)
        return reconstruction, latent_params