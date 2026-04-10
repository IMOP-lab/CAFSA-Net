import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from functools import partial
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from typing import Optional, Sequence, Union
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep
from monai.networks.blocks.dynunet_block import UnetOutBlock
import math
from math import sqrt
from typing import Tuple, Union


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return self._channels_last_norm(x)
        elif self.data_format == "channels_first":
            return self._channels_first_norm(x)
        else:
            raise NotImplementedError("Unsupported data_format: {}".format(self.data_format))

    def _channels_last_norm(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def _channels_first_norm(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class TwoConv(nn.Sequential):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        super().__init__()

        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        feat :int = 96,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class UpCat(nn.Module):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        x_0 = self.upsample(x)

        if x_e is not None:
            # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0)

        return x
    


class SpectralHaarSubbandDecomposer(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        base_kernels = torch.tensor(
            [[[ 0.5,  0.5], [ 0.5,  0.5]],   # LL
             [[ 0.5,  0.5], [-0.5, -0.5]],   # LH
             [[ 0.5, -0.5], [ 0.5, -0.5]],   # HL
             [[ 0.5, -0.5], [-0.5,  0.5]]],  # HH
            dtype=torch.float32
        ).unsqueeze(1)                        # (4,1,2,2)

        weight = base_kernels.repeat(in_channels, 1, 1, 1)  

        self.register_buffer("wave_weight", weight)        
        self.groups = in_channels

        # 可学习缩放
        self.scale = nn.Parameter(torch.ones(4, 1, 1, 1))   

        self.fusion = nn.Conv2d(4 * in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:     
     
        subbands = F.conv2d(
            x, self.wave_weight, stride=2, groups=self.groups
        )                                                   

        ll, lh, hl, hh = subbands.chunk(4, dim=1)
        ll = ll * self.scale[0]
        lh = lh * self.scale[1]
        hl = hl * self.scale[2]
        hh = hh * self.scale[3]

        fused = torch.cat((ll, lh, hl, hh), dim=1)          # (N,4C, H/2,W/2)
        return self.fusion(fused) 



class SpectralSpatialGatedPooling(nn.Module):

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        # Spectral branch (returns C channels):
        self.wave_branch = SpectralHaarSubbandDecomposer(in_channels)

        # Spatial branch: max‑pool & avg‑pool
        self.spatial_pool_max = nn.MaxPool2d(2)
        self.spatial_pool_avg = nn.AvgPool2d(2)

        # ----- dynamic channel‑wise gate (SE‑style) -----
        joint_channels = in_channels * 3  # wave 1C + spatial 2C
        mid_channels = max(in_channels * 3 // reduction, 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(joint_channels, mid_channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid_channels, joint_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Re‑project back to C channels
        self.reprojection = nn.Conv2d(joint_channels, in_channels, 1, bias=False)

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wave = self.wave_branch(x)          # (N, C, H/2, W/2)
        pool_max = self.spatial_pool_max(x) # (N, C, H/2, W/2)
        pool_avg = self.spatial_pool_avg(x) # (N, C, H/2, W/2)

        spatial = torch.cat([pool_max, pool_avg], dim=1)   # 2C
        joint = torch.cat([wave, spatial], dim=1)          # 3C

        gated = joint * self.channel_gate(joint)

        return self.alpha*self.reprojection(gated) + (1 - self.alpha)*pool_max 



class HyperUNetSpectralBranch(nn.Module):
    
    def __init__(self, in_channels: int):
        super().__init__()
        act = nn.Mish()

        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1), act,
            nn.Conv2d(in_channels, in_channels, 3, padding=1), act,
        )
        self.pool1 = SpectralSpatialGatedPooling(in_channels)

        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1), act,
            nn.Conv2d(in_channels, in_channels, 3, padding=1), act,
        )
        self.pool2 = SpectralSpatialGatedPooling(in_channels)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1), act,
            nn.Conv2d(in_channels, in_channels, 3, padding=1), act,
        )

        self.up2 = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1), act,
            nn.Conv2d(in_channels, in_channels, 3, padding=1), act,
        )

        self.up1 = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1), act,
            nn.Conv2d(in_channels, in_channels, 3, padding=1), act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.down1(x)
        p1 = self.pool1(c1)

        c2 = self.down2(p1)
        p2 = self.pool2(c2)

        bn = self.bottleneck(p2)

        u2 = self.up2(bn)
        m2 = self.up_conv2(torch.cat((c2, u2), 1))

        u1 = self.up1(m2)
        m1 = self.up_conv1(torch.cat((c1, u1), 1))
        return m1


class AxialTransformerFusionBlock(nn.Module):
    def __init__(self, c, num_heads=1, dropout=0.0):
        super().__init__()
        assert c % num_heads == 0
        self.c, self.h = c, num_heads
        self.ln_w = nn.LayerNorm(c)
        self.ln_h = nn.LayerNorm(c)
        self.ln_ffn = nn.LayerNorm(c)

        self.qkv_w = nn.Linear(c, 3*c)
        self.qkv_h = nn.Linear(c, 3*c)
        self.proj = nn.Linear(c, c)

        self.ffn = nn.Sequential(
            nn.Linear(c, 2*c), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2*c, c), nn.Dropout(dropout)
        )

    def _attention(self, x, qkv_proj):
        B, L, C = x.shape                       # (B,L,C)
        qkv = qkv_proj(x).reshape(B, L, 3, self.h, C//self.h).permute(2,0,3,1,4)
        q, k, v = qkv                           # (B,h,L,d)
        attn = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(2,1).reshape(B, L, C)
        return self.proj(out)

    def forward(self, x):                       # (N,C,H,W)
        N,C,H_img,W_img = x.shape
        # W‑axis
        x_w = x.permute(0,2,3,1).reshape(-1, W_img, C)
        x_w = x_w + self._attention(self.ln_w(x_w), self.qkv_w)
        x_after_w = x_w.reshape(N, H_img, W_img, C).permute(0,3,1,2)
        # H‑axis
        x_h = x_after_w.permute(0,3,2,1).reshape(-1, H_img, C)
        x_h = x_h + self._attention(self.ln_h(x_h), self.qkv_h)
        x_after_h = x_h.reshape(N, W_img, H_img, C).permute(0,3,2,1)
        # FFN
        y = self.ffn(self.ln_ffn(x_after_h.permute(0,2,3,1))).permute(0,3,1,2)
        return x_after_h + y



class SACF(nn.Module):
        
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % 2 == 0
        half = channels // 2
        self.unet_branch = HyperUNetSpectralBranch(half)
        self.attn_branch = AxialTransformerFusionBlock(half, num_heads)

        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        B, C, D, H, W = x.shape
        x_flat = x.permute(0, 4, 1, 2, 3).contiguous()      
        x_flat = x_flat.view(B * W, C, D, H)                # (B·W, C, D, H)

        c_half = C // 2
        x1, x2 = x_flat[:, :c_half], x_flat[:, c_half:]

        out1 = self.unet_branch(x1)
        out2 = self.attn_branch(x2)

        merged = torch.cat((out1, out2), dim=1)       # (B·W, C, D, H)
        merged = self.fuse(merged).contiguous()  
        out_5d = merged.view(B, W, C, D, H).permute(0, 2, 3, 4, 1)
       

        return out_5d



class AsCoT_Former(nn.Module):
      
    def __init__(
        self,
        in_channels: int,
        patch_size: int = 3,
        embed_ratio: int = 2,
        num_heads: int = 1,
        attn_drop: float = 0.0,
        use_d_fusion: bool = True,
    ) -> None:
        super().__init__()
        if in_channels % num_heads != 0:
            raise ValueError("`in_channels` must be divisible by `num_heads`.")

        # ---------- patch‑embed / unembed ----------
        self.patch_size = patch_size
        if patch_size == 1:
            self.patch_embed = nn.Identity()
            self.patch_unembed = nn.Identity()
            self.embed_dim = in_channels
        else:
            self.embed_dim = embed_ratio * in_channels
            self.patch_embed = nn.Conv2d(
                in_channels, self.embed_dim,
                kernel_size=patch_size, stride=patch_size
            )
            self.patch_unembed = nn.ConvTranspose2d(
                self.embed_dim, in_channels,
                kernel_size=patch_size, stride=patch_size
            )

        mha_args = dict(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )

        # ---------- Axial attentions: H / W first, then D ----------
        self.attn_h = nn.MultiheadAttention(**mha_args)  # (B*Wp*D, Hp, E)
        self.attn_w = nn.MultiheadAttention(**mha_args)  # (B*D*Hp, Wp, E)
        self.attn_d = nn.MultiheadAttention(**mha_args)  # (B*Wp*Hp, D, E)

        # ---------- LayerNorms ----------
        self.ln_h1 = nn.LayerNorm(self.embed_dim)
        self.ln_h2 = nn.LayerNorm(self.embed_dim)
        self.ln_w1 = nn.LayerNorm(self.embed_dim)
        self.ln_w2 = nn.LayerNorm(self.embed_dim)
        self.ln_d1 = nn.LayerNorm(self.embed_dim)
        self.ln_d2 = nn.LayerNorm(self.embed_dim)

        # ---------- Feed-forward nets ----------
        def _ffn():
            return nn.Sequential(
                nn.Linear(self.embed_dim, 4*self.embed_dim),
                nn.GELU(),
                nn.Linear(4*self.embed_dim, self.embed_dim),
            )

        self.ffn_h = _ffn()
        self.ffn_w = _ffn()
        self.ffn_d = _ffn()

        
        self.hw_gate = nn.Parameter(torch.tensor(0.5))  

        # ------------- Frequency‑domain enhancement -------------
        self.freq_mag_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0, groups=in_channels, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=in_channels),
            nn.GELU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False),
        )

        self.freq_phase_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0, groups=in_channels, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=in_channels),
            nn.GELU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False),
        )

        self.freq_gate = nn.Parameter(torch.tensor(0.5))

        # ---------- Optional d‑axis depth‑wise fusion ----------
        self.use_d_fusion = use_d_fusion
        if use_d_fusion:
            self.d_fuse = nn.Sequential(
                nn.Conv3d(
                    in_channels, in_channels,
                    kernel_size=(3, 1, 1), padding=(1, 0, 0),
                    groups=in_channels, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=in_channels),
                nn.SiLU(inplace=True),
            )

    # -------------------- forward ------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, D, H, W)
        """
        ori = x
        B, C, D, H, W = x.shape

        # --------- Patch‑embed along (H, W) per slice ---------
        if self.patch_size > 1:
            if H % self.patch_size or W % self.patch_size:
                raise ValueError("`patch_size` must divide both H and W.")
            x_pe = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)  # (B*D, C, H, W)
            x_pe = self.patch_embed(x_pe)                             # (B*D, E, Hp, Wp)
            Hp, Wp = x_pe.shape[-2:]
            x = (
                x_pe.view(B, D, self.embed_dim, Hp, Wp)
                    .permute(0, 2, 1, 3, 4)                           # (B, E, D, Hp, Wp)
                    .contiguous()
            )
        else:
            Hp, Wp = H, W
            x = x  # (B, C, D, H, W) -> treat C as E

        E = self.embed_dim

        # =========================================================
        # 1) H / W 
        # =========================================================

        # ---- H ：(B*Wp*D, Hp, E) ----
        x_h = x.permute(0, 4, 2, 3, 1).reshape(B * Wp * D, Hp, E)
        h = x_h + self.attn_h(self.ln_h1(x_h), self.ln_h1(x_h), self.ln_h1(x_h))[0]
        h = h + self.ffn_h(self.ln_h2(h))
        h = (
            h.view(B, Wp, D, Hp, E)
             .permute(0, 4, 2, 3, 1)            # (B, E, D, Hp, Wp)
             .contiguous()
        )

        # ---- W ：(B*D*Hp, Wp, E) ----
        x_w = x.permute(0, 2, 3, 4, 1).reshape(B * D * Hp, Wp, E)
        w = x_w + self.attn_w(self.ln_w1(x_w), self.ln_w1(x_w), self.ln_w1(x_w))[0]
        w = w + self.ffn_w(self.ln_w2(w))
        w = (
            w.view(B, D, Hp, Wp, E)
             .permute(0, 4, 1, 2, 3)            # (B, E, D, Hp, Wp)
             .contiguous()
        )

        
        alpha = torch.sigmoid(self.hw_gate)
        x_hw = alpha * h + (1.0 - alpha) * w      # (B, E, D, Hp, Wp)

        # =========================================================
        # 2) D：(B*Wp*Hp, D, E)
        # =========================================================
        x_d = x_hw.permute(0, 4, 3, 2, 1).reshape(B * Wp * Hp, D, E)
        d = x_d + self.attn_d(self.ln_d1(x_d), self.ln_d1(x_d), self.ln_d1(x_d))[0]
        d = d + self.ffn_d(self.ln_d2(d))
        x_d = (
            d.view(B, Wp, Hp, D, E)
             .permute(0, 4, 3, 2, 1)             # (B, E, D, Hp, Wp)
             .contiguous()
        )

        # ----------- patch‑unembed -----------
        if self.patch_size > 1:
            x_2d = x_d.permute(0, 2, 1, 3, 4).reshape(B * D, E, Hp, Wp)
            x_out = self.patch_unembed(x_2d)                      # (B*D, C, H, W)
            x_out = (
                x_out.view(B, D, C, H, W)
                      .permute(0, 2, 1, 3, 4)                     # (B, C, D, H, W)
                      .contiguous()
            )
        else:
            x_out = x_d  

        # ---------- frequency‑domain enhancement ----------
        x_fft = torch.fft.fftn(x_out, dim=(-3, -2, -1))
        mag, phase = x_fft.abs(), torch.angle(x_fft)
        mag = mag + self.freq_mag_conv(mag)
        phase = phase + self.freq_phase_conv(phase)
        x_ifft = torch.fft.ifftn(torch.polar(mag, phase), dim=(-3, -2, -1)).real
        gate = torch.sigmoid(self.freq_gate)
        x_enh = gate * x_ifft + (1.0 - gate) * x_out

        # optional d‑axis depth‑wise fusion
        if self.use_d_fusion:
            x_enh = self.d_fuse(x_enh)

        return x_enh + ori


    
class CAFSANet(nn.Module):
    
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        
        features: Sequence[int] = (24, 48, 96, 192, 384, 24),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
    ):
        super().__init__()
        
        if dimensions is not None:
            spatial_dims = dimensions
        fea = ensure_tuple_rep(features, 6)


        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout,feat=96)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout,feat=48)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout,feat=24)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout,feat=12)

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

        self.sacf_0 = SACF(channels=fea[0], num_heads=1)
        self.sacf_1 = SACF(channels=fea[1], num_heads=1)
        self.sacf_2 = SACF(channels=fea[2], num_heads=1)
        self.sacf_3 = SACF(channels=fea[3], num_heads=1)
        
        self.ascot_former_0 = AsCoT_Former(in_channels=fea[0], num_heads=2)
        self.ascot_former_1 = AsCoT_Former(in_channels=fea[1], num_heads=2)
        self.ascot_former_2 = AsCoT_Former(in_channels=fea[2], num_heads=2)
        self.ascot_former_3 = AsCoT_Former(in_channels=fea[3], num_heads=2) 

    def forward(self, x: torch.Tensor):
                
        x0 = self.conv_0(x)
        x0 = self.ascot_former_0(x0)
        x1 = self.down_1(x0) 
        x1 = self.ascot_former_1(x1)             
        x2 = self.down_2(x1)
        x2 = self.ascot_former_2(x2)        
        x3 = self.down_3(x2)
        x3 = self.ascot_former_3(x3)
        x4 = self.down_4(x3)
             
        x0 = self.sacf_0(x0) 
        x1 = self.sacf_1(x1) 
        x2 = self.sacf_2(x2) 
        x3 = self.sacf_3(x3)
        
        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)
        
        logits = self.final_conv(u1)

        return logits