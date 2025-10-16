import torch
import torch.nn as nn
from ddpm_model.models.UNetBlocks import (
    DownSamplingBlock,
    BottleNeck,
    UpSamplingBlock,
)
from ddpm_model.models.TimeEmbedding import TimeEmbedding, time_embedding_fun

class UNet(nn.Module):
    """
    UNet-like architecture with attention and time embedding for diffusion models.

    The architecture consists of:
    1. A series of downsampling blocks
    2. A bottleneck block
    3. A series of upsampling blocks with skip connections

    Args:
        in_chanels: Number of input channels
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

        # Channel configurations
        self.down_channels = self.UnetParams["down_channels"]
        self.time_emb_dim = self.UnetParams["time_emb_dim"]
        self.down_sample = self.UnetParams["down_sample"]
        self.in_channels = self.UnetParams["im_channels"]
        self.mid_channels = self.UnetParams["mid_channels"]
        self.num_down_layers = self.UnetParams["num_down_layers"]
        self.num_mid_layers = self.UnetParams["num_mid_layers"]
        self.num_up_layers = self.UnetParams["num_up_layers"]
        self.attns = self.UnetParams["attn_down"]

        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1

        # Time embedding projection
        self.time_proj = TimeEmbedding(self.time_emb_dim, True)

        self.up_sample = list(reversed(self.down_sample))

        # Initial convolution
        self.conv_in = nn.Conv2d(
            self.in_channels, self.down_channels[0], kernel_size=3, padding=1
        )

        # Downsampling blocks
        self.down_sampling = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.down_sampling.append(
                DownSamplingBlock(
                    in_channels=self.down_channels[i],
                    out_channels=self.down_channels[i + 1],
                    time_emb_dim=self.time_emb_dim,
                    down_sample=self.down_sample[i],
                    num_layers=self.num_down_layers,
                    use_attn=self.attns[i],
                )
            )

        # Bottleneck blocks
        self.bottleneck = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.bottleneck.append(
                BottleNeck(
                    self.mid_channels[i],
                    self.mid_channels[i + 1],
                    self.time_emb_dim,
                    num_layers=self.num_mid_layers,
                )
            )

        # Upsampling blocks
        # Build upsampling path with correct channel flow and skip concatenation
        self.up_sampling = nn.ModuleList([])
        current_in = self.mid_channels[-1]
        for i in reversed(range(len(self.down_channels) - 1)):
            skip_ch = self.down_channels[i]
            pre_in = current_in  # channels before concatenation
            out_ch = self.down_channels[i - 1] if i != 0 else 16
            self.up_sampling.append(
                UpSamplingBlock(
                    in_channels=pre_in,
                    out_channels=out_ch,
                    skip_channels=skip_ch,
                    time_emb_dim=self.time_emb_dim,
                    up_sample=self.down_sample[i],
                    num_layers=self.num_up_layers,
                    use_attn=True,
                )
            )
            current_in = out_ch

        # Output layers
        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, self.in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t):
        """
        Forward pass of the UNet.

        Args:
            x: Input tensor of shape (B, C, H, W)
            t: Timestep for time embedding

        Returns:
            Output tensor of same shape as input
        """
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        # B x C x H x W
        out = self.conv_in(x)
        # B x C1 x H x W

        # t_emb -> B x t_emb_dim
        B = x.shape[0]
        # Support scalar int, 0-dim tensor, or per-sample (B,) tensor for timesteps
        if isinstance(t, torch.Tensor):
            if t.dim() == 0:
                t_vec = t.expand(B).to(device=x.device, dtype=torch.float32)
            else:
                assert t.shape[0] == B, "t must be scalar or have shape (B,)"
                t_vec = t.to(device=x.device, dtype=torch.float32)
        else:
            t_vec = torch.full((B,), float(t), device=x.device, dtype=torch.float32)
        t_emb = time_embedding_fun(t_vec, self.time_emb_dim)
        t_emb = self.time_proj(t_emb)

        down_outs = []

        # Downsampling path
        for idx, down in enumerate(self.down_sampling):
            down_outs.append(out)
            out = down(out, t_emb)
        # down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
        # out B x C4 x H/4 x W/4

        # Bottleneck
        for mid in self.bottleneck:
            out = mid(out, t_emb)
        # out B x C3 x H/4 x W/4

        # Upsampling path
        for up in self.up_sampling:
            down_out = down_outs.pop()
            out = up(out, t_emb, down_out)
            # out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]

        # Final output
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        # out B x C x H x W
        return out