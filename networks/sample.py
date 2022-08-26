import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class PatchTransformerEncoder(nn.Module):
    def __init__(self, in_channels, patch_size=32, embedding_dim=64,
                 num_heads=4, num_layers=4, n_query_channels=16):
        super(PatchTransformerEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_query_channels=n_query_channels

        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)  # takes shape S,N,E

        self.embedding_convPxP = nn.Conv2d(
            in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0
        )
        self.positional_encodings = nn.Parameter(torch.rand(500, embedding_dim), requires_grad=True)

        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        batch, _, height, width = x.shape

        embeddings = self.embedding_convPxP(x).flatten(2)  # .shape = n,c,s = n, embedding_dim, s
        # embeddings = nn.functional.pad(embeddings, (1,0))  # extra special token at start ?
        embeddings = embeddings + self.positional_encodings[:embeddings.shape[2], :].T.unsqueeze(0)

        # change to S,N,E format required by transformer
        embeddings = embeddings.permute(2, 0, 1)
        tgt = self.transformer_encoder(embeddings)  # .shape = S, N, E
        queries = tgt[:self.n_query_channels, ...]

        feat = self.conv3x3(x)
        attn_feat = torch.matmul(
            feat.view(batch, self.embedding_dim, height * width).permute(0, 2, 1),  # [N, H*W, embedding_dim]
            queries.permute(1, 2, 0)  # [N, embedding_dim, n_query_channels, ]
        ).permute(0, 2, 1).view(batch, self.n_query_channels, height, width)

        return attn_feat

class AdaptiveBins(nn.Module):
    def __init__(self, in_channels, n_bins=48, n_query_channels=16,
                 patch_size=32, embedding_dim=64, num_heads=4, norm='linear'):
        super(AdaptiveBins, self).__init__()

        self.norm = norm
        self.n_query_channels = n_query_channels

        self.patch_transformer = PatchTransformerEncoder(
            in_channels, patch_size, embedding_dim,
            num_heads, num_layers=4, n_query_channels=n_query_channels
        )
        self.regressor = nn.Sequential(
            nn.Conv2d(n_query_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, n_bins, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        # x: features, (N C H W)
        tgt = self.patch_transformer(x)
        bins = self.regressor(tgt)

        if self.norm == 'linear':
            eps = 0.1
            bins = torch.relu(bins)
            bins = bins + eps
            bins = bins / bins.sum(dim=1, keepdim=True)
        if self.norm == 'softmax':
            bins = torch.softmax(bins)
        if self.norm == 'sigmoid':
            bins = torch.sigmoid(bins)
            bins = bins / bins.sum(dim=1, keepdim=True)

        return bins

class GaussBins(nn.Module):
    def __init__(self):
        super(GaussBins, self).__init__()
        self.loc = 0.0
        self.scale = 1.0
        self.gauss = Normal(self.loc, self.scale)

    @torch.no_grad()
    def forward(self, last_depth, ndepth, depth_interval_pixel, cost_reg):
        dtype = last_depth.dtype
        device = last_depth.device

        cost_reg = cost_reg.squeeze(1)  # (B, 1, D, H, W) -> (B, D, H, W)
        prob_volume = F.softmax(cost_reg, dim=1)
        entropy = torch.div(
            torch.sum(prob_volume * prob_volume.clamp(1e-9, 1.).log(), dim=1, keepdim=True),
            -math.log(ndepth)
        )
        ranges = 2 * (self.gauss.cdf(1 + entropy * self.scale) - 0.5)
        bin_edges = ranges / ndepth * torch.arange(
            0, ndepth + 1, device=device, dtype=dtype
        ).reshape(1, -1, 1, 1) + self.gauss.cdf(-1 - entropy * self.scale)
        bin_edges = self.gauss.icdf(bin_edges)
        bin_widths = bin_edges[:, 1:] - bin_edges[:, :-1]
        bin_widths_normed = bin_widths / bin_widths.sum(dim=1, keepdim=True)

        last_depth_min = last_depth - ndepth / 2 * depth_interval_pixel
        last_depth_max = last_depth + ndepth / 2 * depth_interval_pixel  # (B, H, W)
        bin_widths = (last_depth_max - last_depth_min).unsqueeze(1) * bin_widths_normed
        bin_edges = torch.cat([last_depth_min.unsqueeze(1), bin_widths], dim=1)
        bin_edges = torch.cumsum(bin_edges, dim=1)

        depth_samples = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        # [B, D, H, W], [B, D, H, W]
        return depth_samples, bin_widths

class MixedSamples(nn.Module):
    def __init__(self, in_channel, ndepth, mixed=True):
        '''
        Args:
            mixed: mixed uniform sample and adaptive sample
        '''
        super(MixedSamples, self).__init__()
        
        self.num_adaptive_bins = ndepth // 4 if mixed else ndepth
        self.num_uniform_bins = ndepth - self.num_adaptive_bins
        if mixed:
            assert (self.num_uniform_bins > 1)
            assert (self.num_uniform_bins + self.num_adaptive_bins) == ndepth
        
        self.adaptive_bins_layer = AdaptiveBins(in_channel, self.num_adaptive_bins)
        
    def forward(self, last_depth, shape, ref_feat=None):
        dtype = last_depth.dtype
        device = last_depth.device
        batch = last_depth.size(0)
        height, width = shape[0], shape[1]
        
        last_depth_min = last_depth[:, 0]
        last_depth_max = last_depth[:, -1]  # (B,)
        
        # adaptive
        adaptive_bin_widths_normed = self.adaptive_bins_layer(ref_feat)  # (B, ndepth, H, W)
        adaptive_bin_widths = (last_depth_max - last_depth_min).view(batch, 1, 1, 1) * adaptive_bin_widths_normed
        adaptive_bin_edges = torch.cat(
            [last_depth_min.view(batch, 1, 1, 1).repeat(1, 1, height, width), adaptive_bin_widths],
            dim=1
        )
        adaptive_bin_edges = torch.cumsum(adaptive_bin_edges, dim=1)
        adaptive_depth_samples = 0.5 * (adaptive_bin_edges[:, :-1] + adaptive_bin_edges[:, 1:])
        
        if self.num_uniform_bins == 0:
                return adaptive_depth_samples, adaptive_bin_edges
        
        # uniform
        new_interval = (last_depth_max - last_depth_min) / (self.num_uniform_bins - 1)
        uniform_depth_samples = last_depth_min.unsqueeze(1) + (
            torch.arange(
                0, self.num_uniform_bins, device=device, dtype=dtype, requires_grad=False,
            ).reshape(1, -1) * new_interval.unsqueeze(1)
        )  # (B, D)
        uniform_depth_samples = (
            uniform_depth_samples.view(batch, self.num_uniform_bins, 1, 1).repeat(1, 1, height, width)
        )  # (B, D, H, W)
        
        # merged
        depth_samples = torch.cat([adaptive_depth_samples, uniform_depth_samples], dim=1)
        depth_samples, _ = torch.sort(depth_samples, dim=1)
        bin_widths = depth_samples[:, 1:] - depth_samples[:, :-1]
        bin_widths = torch.cat([bin_widths, bin_widths[:, -1:]], dim=1)
        
        return depth_samples, bin_widths

class AdaptiveSamples(nn.Module):
    def __init__(self, in_channels, ndepths, norm='linear'):
        super(AdaptiveSamples, self).__init__()

        assert norm in ['linear', 'softmax', 'sigmoid']
        
        self.num_adaptive_bins = ndepths[0] // 4
        self.num_uniform_bins = ndepths[0] - self.num_adaptive_bins

        self.gauess_bins_layer = GaussBins()
        self.mixed_bins_layer = MixedSamples(in_channels[0], ndepths[0], mixed=True)

    def forward(self, last_depth, ndepth, depth_interval_pixel, shape, ref_feat=None, cost_reg=None):
        if last_depth.dim() == 2:
            # Merge adaptive sampling and uniform sampling
            depth_samples, bin_widths = self.mixed_bins_layer(
                last_depth, shape, ref_feat
            )
        else:
            depth_samples, bin_widths = self.gauess_bins_layer(
                last_depth, ndepth, depth_interval_pixel, cost_reg
            )
        # [B, D, H, W], [B, D, H, W]
        return depth_samples, bin_widths


def get_cur_uniform_samples(last_depth, ndepth, depth_interval_pixel):
    '''
    Args:
        cur_depth: (B, H, W)
    Returns:
        depth_range_values: (B, D, H, W)
    '''
    dtype = last_depth.dtype
    device = last_depth.device
    last_depth_min = last_depth - ndepth / 2 * depth_interval_pixel
    last_depth_max = last_depth + ndepth / 2 * depth_interval_pixel  # (B, H, W)
    new_interval = (last_depth_max - last_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = last_depth_min.unsqueeze(1) + (
        torch.arange(
            0, ndepth, device=device, dtype=dtype, requires_grad=False
        ).reshape(1, -1, 1, 1) * new_interval.unsqueeze(1)
    )
    bin_widths = new_interval.unsqueeze(1).repeat(1, ndepth, 1, 1)
    # [B, D, H, W], [B, D, H, W]
    return depth_range_samples, bin_widths

def get_uniform_samples(last_depth, ndepth, depth_interval_pixel, shape=None, **kwargs):
    '''
    Args:
        last_depth: (B, H, W) or (B, D)
    Returns:
        bin_widths: (B, D, H, W)
        depth_samples: (B, D, H, W)
    '''
    batch = last_depth.size(0)
    height, width = shape[0], shape[1]
    dtype = last_depth.dtype
    device = last_depth.device

    if last_depth.dim() == 2:
        last_depth_min = last_depth[:, 0]
        last_depth_max = last_depth[:, -1]  # (B,)
        new_interval = (last_depth_max - last_depth_min) / (ndepth - 1)

        depth_samples = last_depth_min.unsqueeze(1) + (
            torch.arange(
                0, ndepth, device=device, dtype=dtype, requires_grad=False,
            ).reshape(1, -1) * new_interval.unsqueeze(1)
        )  # (B, D)
        depth_samples = (
            depth_samples.view(batch, ndepth, 1, 1).repeat(1, 1, height, width)
        )  # (B, D, H, W)
        bin_widths = new_interval.view(batch, 1, 1, 1).repeat(1, ndepth, height, width)
    else:
        depth_samples, bin_widths = get_cur_uniform_samples(
            last_depth, ndepth, depth_interval_pixel
        )
    # [B, D, H, W], [B, D, H, W]
    return depth_samples, bin_widths


def get_depth_samples(in_channels, ndepths, mode='adaptive'):
    assert mode in ['adaptive', 'uniform']
    if mode == 'adaptive':
        return AdaptiveSamples(in_channels, ndepths, norm='linear')
    else:
        return get_uniform_samples


if __name__ == '__main__':

    min_depth = torch.ones(2, 1) * 100.0
    max_depth = torch.ones(2, 1) * 200.0
    last_depth = torch.cat([min_depth, max_depth], dim=1)

    # last_depth = torch.ones(2, 128, 128) * 150

    depth_interval_pixel = 2.5
    ref_feat = torch.rand(2, 32, 128, 128)
    cost_reg = torch.rand(2, 1, 16, 128, 128)
    depth_sampling = get_depth_samples([32, 16, 8], [16, 8, 8], mode='adaptive')

    depth_range_samples, interval = depth_sampling(
        last_depth=last_depth,
        ndepth=16,
        depth_interval_pixel=depth_interval_pixel,
        shape=[128, 128],  # only for first stage
        ref_feat = ref_feat,
        cost_reg = cost_reg
    )

    print(interval[0, :, 64, 64])
    print(depth_range_samples[0, :, 64, 64])
    print(depth_range_samples.shape)