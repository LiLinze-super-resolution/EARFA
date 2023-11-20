import math

import torch
from torch import nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


class ShiftConv2d1(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d1, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)
        self.n_div = 5
        g = inp_channels // self.n_div
        self.weight[0 * g:1 * g, 0, 1, 2] = 1.0  ## left
        self.weight[1 * g:2 * g, 0, 1, 0] = 1.0  ## right
        self.weight[2 * g:3 * g, 0, 2, 1] = 1.0  ## up
        self.weight[3 * g:4 * g, 0, 0, 1] = 1.0  ## down
        self.weight[4 * g:, 0, 1, 1] = 1.0  ## identity

        self.conv1x1 = nn.Conv2d(inp_channels, out_channels, 1)

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.inp_channels)
        y = self.conv1x1(y)
        return y


class ShiftConv2d(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        self.shift_conv = ShiftConv2d1(inp_channels, out_channels)

    def forward(self, x):
        y = self.shift_conv(x)
        return y


class ECA(nn.Module):

    def __init__(self, num_feat=64, middle_channels=4):
        super(ECA, self).__init__()

        self.encoder = nn.Conv2d(in_channels=num_feat, out_channels=middle_channels, kernel_size=1)

        # there will be calculating entropy

        self.decoder = nn.Conv2d(in_channels=middle_channels, out_channels=num_feat, kernel_size=1)

    def forward(self, x):
        identity = x.clone()

        x1 = self.encoder(x)

        x_var = torch.var(x1, keepdim=True, dim=(2, 3))
        entropy = torch.log(2 * math.pi * math.e * x_var) / 2.0

        attention = self.decoder(entropy)
        standard_attention = torch.sigmoid(attention)

        return identity * standard_attention


class SLKA(nn.Module):

    def __init__(self,
                 num_feat,
                 dw_size=7,
                 dw_di_size=9):
        super(SLKA, self).__init__()
        self.pointwise = ShiftConv2d(num_feat, num_feat)
        self.depthwise = nn.Conv2d(num_feat, num_feat, dw_size, padding=(dw_size - 1) // 2, groups=num_feat)
        self.depthwise_dilated = nn.Conv2d(num_feat, num_feat, dw_di_size, stride=1,
                                           padding=((dw_di_size - 1) * 3) // 2,
                                           groups=num_feat,
                                           dilation=3)

    def forward(self, x):
        identity = x.clone()
        attention = self.pointwise(x)
        attention = self.depthwise(attention)
        attention = self.depthwise_dilated(attention)
        return identity * attention


class SGFN(nn.Module):
    def __init__(self,
                 num_feat,
                 upscale_sgfn=2,
                 dw_size_sgfn=5):
        super(SGFN, self).__init__()
        self.num_feat = num_feat

        self.sgfn_feat = self.num_feat * upscale_sgfn
        self.encoder_1 = nn.Conv2d(self.num_feat, self.sgfn_feat, 1)
        self.dwconv = nn.Conv2d(in_channels=self.sgfn_feat // 2, out_channels=self.sgfn_feat // 2,
                                kernel_size=dw_size_sgfn,
                                padding=(dw_size_sgfn - 1) // 2, groups=self.sgfn_feat // 2)
        self.decoder_1 = nn.Conv2d(self.sgfn_feat // 2, num_feat, 1)

    def forward(self, x):
        sgfn_feat = self.encoder_1(x)
        sgfn_feat_res, sgfn_feat_dw = torch.split(sgfn_feat, self.sgfn_feat // 2, dim=1)
        ffn_feat_dw = self.dwconv(sgfn_feat_dw)
        sgfn_feat = ffn_feat_dw * sgfn_feat_res

        out = self.decoder_1(sgfn_feat)

        return out


class EDCB(nn.Module):
    def __init__(self,
                 num_feat,
                 upscale_attn,
                 downscale_eca,
                 upscale_sgfn=2,
                 dw_size_sgfn=5):
        super(EDCB, self).__init__()
        self.num_feat = num_feat
        # layernorm
        self.layer_norm_pre = nn.LayerNorm(self.num_feat)

        # attention
        self.attn_feat = self.num_feat * upscale_attn
        self.encoder = nn.Conv2d(self.num_feat, self.attn_feat, 1)
        self.eca = ECA(self.attn_feat, self.attn_feat // downscale_eca)
        self.decoder = nn.Conv2d(self.attn_feat, self.num_feat, 1)

        # LayerNorm
        self.layer_norm_aft = nn.LayerNorm(self.num_feat)

        # SGFN
        self.sgfn = SGFN(self.num_feat, upscale_sgfn, dw_size_sgfn)

        # activation
        self.gelu = nn.GELU()

    def forward(self, x):
        identity = x.clone()
        ln_x = self.layer_norm_pre(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

        attn_feat = self.gelu(self.encoder(ln_x))
        attn_out = self.eca(attn_feat)
        attn_out = self.decoder(attn_out) + identity

        middle_feat = attn_out.clone()
        ln_feat = self.layer_norm_aft(attn_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

        out = self.sgfn(ln_feat)

        return out + middle_feat


class EDSB(nn.Module):
    def __init__(self,
                 num_feat,
                 upscale_attn=2,
                 dw_size=7,
                 dw_di_size=9,
                 upscale_sgfn=2,
                 dw_size_sgfn=5):
        super(EDSB, self).__init__()
        self.num_feat = num_feat
        # layernorm
        self.layer_norm_pre = nn.LayerNorm(self.num_feat)

        # attention
        self.attn_feat = self.num_feat * upscale_attn
        self.encoder = nn.Conv2d(self.num_feat, self.attn_feat, 1)
        self.slka = SLKA(self.attn_feat, dw_size=dw_size, dw_di_size=dw_di_size)
        self.decoder = nn.Conv2d(self.attn_feat, self.num_feat, 1)

        # LayerNorm
        self.layer_norm_aft = nn.LayerNorm(self.num_feat)

        # SGFN
        self.sgfn = SGFN(self.num_feat, upscale_sgfn, dw_size_sgfn)

        # activation
        self.gelu = nn.GELU()

    def forward(self, x):
        identity = x.clone()
        ln_x = self.layer_norm_pre(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

        attn_feat = self.gelu(self.encoder(ln_x))
        attn_out = self.gelu(self.slka(attn_feat))
        attn_out = self.decoder(attn_out) + identity

        middle_feat = attn_out.clone()
        ln_feat = self.layer_norm_aft(attn_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

        out = self.sgfn(ln_feat)

        return out + middle_feat


class EDTB(nn.Module):

    def __init__(self,
                 num_feat,
                 upscale_attn=2,
                 upscale_sgfn=2,
                 dw_size=7,
                 dw_di_size=9,
                 downscale_eca=8,
                 dw_size_sgfn=5):
        super(EDTB, self).__init__()

        # spatial attention block
        self.edsb = EDSB(num_feat=num_feat,
                         upscale_attn=upscale_attn,
                         dw_size=dw_size,
                         dw_di_size=dw_di_size,
                         upscale_sgfn=upscale_sgfn,
                         dw_size_sgfn=dw_size_sgfn)

        # channel attention block
        self.edcb = EDCB(num_feat=num_feat,
                         upscale_attn=upscale_attn,
                         downscale_eca=downscale_eca,
                         upscale_sgfn=upscale_sgfn,
                         dw_size_sgfn=dw_size_sgfn)

    def forward(self, x):
        out_edsb = self.edsb(x)
        out = self.edcb(out_edsb)

        return out


class Upsample(nn.Sequential):

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def UpsampleOneStep(in_channels, out_channels, upscale_factor=4):
    conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])


class PixelShuffleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, upscale_factor=4):
        super().__init__()
        num_feat = 64
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(in_channels, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(upscale_factor, num_feat)
        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))
        return x


@ARCH_REGISTRY.register()
class EARFA(nn.Module):
    """
    Encoder-Decoder Network
    a lightweight SR model
    """

    def __init__(self,
                 num_feat=60,
                 num_blocks=12,
                 upscale_attn=2,
                 upscale_sgfn=2,
                 dw_size=5,
                 dw_di_size=7,
                 downscale_eca=8,
                 dw_size_sgfn=5,
                 upsampler='pixelshuffledirect',
                 img_range=1,
                 upscale_sr=4
                 ):
        super(EARFA, self).__init__()
        self.num_feat = num_feat
        self.img_range = img_range
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        # shallow feature process
        self.simple_encoder = nn.Conv2d(3, self.num_feat, 3, padding=1)

        # deep feature process
        self.pipeline = nn.Sequential(
            *[EDTB(num_feat=self.num_feat,
                   upscale_attn=upscale_attn,
                   upscale_sgfn=upscale_sgfn,
                   dw_size=dw_size,
                   dw_di_size=dw_di_size,
                   dw_size_sgfn=dw_size_sgfn,
                   downscale_eca=downscale_eca)
              for _ in range(num_blocks)]
        )

        self.after_pipe = nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 1)

        self.gelu = nn.GELU()

        if upsampler == 'pixelshuffledirect':
            self.upsampler = UpsampleOneStep(self.num_feat, 3, upscale_factor=upscale_sr)
        elif upsampler == 'pixelshuffle':
            self.upsampler = PixelShuffleBlock(self.num_feat, 3, upscale_factor=upscale_sr)
        else:
            raise NotImplementedError("Check the Upsampler. None or not support yet.")

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        out_fea = self.gelu(self.simple_encoder(x))
        pipeline_out = self.after_pipe(self.pipeline(out_fea)) + out_fea

        out = self.upsampler(pipeline_out)

        out = out / self.img_range + self.mean
        return out
