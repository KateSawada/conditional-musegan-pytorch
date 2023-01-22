import torch
import torch.nn.functional as F


class GeneraterBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.transconv = torch.nn.ConvTranspose3d(in_dim, out_dim, kernel, stride)
        self.batchnorm = torch.nn.BatchNorm3d(out_dim)

    def forward(self, x):
        x = self.transconv(x)
        x = self.batchnorm(x)
        return torch.nn.functional.relu(x)

class FinalGeneraterBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.transconv = torch.nn.ConvTranspose3d(in_dim, out_dim, kernel, stride)
        self.batchnorm = torch.nn.BatchNorm3d(out_dim)

    def forward(self, x):
        x = self.transconv(x)
        x = self.batchnorm(x)
        return x


def create_generator_from_config(config):
    n_tracks = config.n_tracks  # number of tracks
    n_pitches = config.n_pitches  # number of pitches
    n_measures = config.n_measures  # number of measures per sample
    beat_resolution = config.beat_resolution  # temporal resolution of a beat (in timestep)
    latent_dim = config.latent_dim
    conditioning = config.conditioning
    conditioning_dim = config.conditioning_dim

    measure_resolution = 4 * beat_resolution
    generator = Generator(latent_dim, n_tracks, n_measures, measure_resolution, n_pitches, conditioning, conditioning_dim)

    return generator


class Generator(torch.nn.Module):
    """A convolutional neural network (CNN) based generator. The generator takes
    as input a latent vector and outputs a fake sample."""
    def __init__(self, latent_dim, n_tracks, n_measures, measure_resolution, n_pitches,
                 conditioning=False, conditioning_dim=0):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_tracks = n_tracks
        self.n_measures = n_measures
        self.measure_resolution = measure_resolution
        self.n_pitches = n_pitches
        self.conditioning = conditioning
        self.conditioning_dim = conditioning_dim
        if (self.conditioning):
            self.transconv0 = GeneraterBlock(latent_dim + conditioning_dim, 256, (4, 1, 1), (4, 1, 1))
        else:
            self.transconv0 = GeneraterBlock(latent_dim, 256, (4, 1, 1), (4, 1, 1))
        self.transconv1 = GeneraterBlock(256, 128, (1, 4, 1), (1, 4, 1))
        self.transconv2 = GeneraterBlock(128, 64, (1, 1, 4), (1, 1, 4))
        self.transconv3 = GeneraterBlock(64, 32, (1, 1, 3), (1, 1, 1))
        self.transconv4 = torch.nn.ModuleList([
            GeneraterBlock(32, 16, (1, 4, 1), (1, 4, 1))
            for _ in range(n_tracks)
        ])
        self.transconv5 = torch.nn.ModuleList([
            FinalGeneraterBlock(16, 1, (1, 1, 12), (1, 1, 12))
            for _ in range(n_tracks)
        ])


    def forward(self, x):
        # conditioning
        if (self.conditioning):
            x, condition = x
            condition = condition.view(-1, self.conditioning_dim)
            shape = list(x.shape)
            shape[1] = self.conditioning_dim
            # match shape of x and condition excluding dim 1
            condition = condition.expand(shape)
            x = torch.cat([x, condition], 1)
        x = x.view(-1, self.latent_dim + self.conditioning_dim, 1, 1, 1)
        x = self.transconv0(x)
        x = self.transconv1(x)
        x = self.transconv2(x)
        x = self.transconv3(x)
        x = [transconv(x) for transconv in self.transconv4]
        x = torch.cat([transconv(x_) for x_, transconv in zip(x, self.transconv5)], 1)
        x = x.view(-1, self.n_tracks, self.n_measures * self.measure_resolution, self.n_pitches)
        x = torch.sigmoid(x)
        return x


class LayerNorm(torch.nn.Module):
    """An implementation of Layer normalization that does not require size
    information. Copied from https://github.com/pytorch/pytorch/issues/1959."""
    def __init__(self, n_features, eps=1e-5, affine=True):
        super().__init__()
        self.n_features = n_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = torch.nn.Parameter(torch.Tensor(n_features).uniform_())
            self.beta = torch.nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


class DiscriminatorBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.transconv = torch.nn.Conv3d(in_dim, out_dim, kernel, stride)
        self.layernorm = LayerNorm(out_dim)

    def forward(self, x):
        x = self.transconv(x)
        x = self.layernorm(x)
        return torch.nn.functional.leaky_relu(x)

class Discriminator(torch.nn.Module):
    """A convolutional neural network (CNN) based discriminator. The
    discriminator takes as input either a real sample (in the training data) or
    a fake sample (generated by the generator) and outputs a scalar indicating
    its authentity.
    """
    def __init__(self, n_tracks, n_measures, measure_resolution, n_pitches,
                 conditioning=False, conditioning_dim=0):
        self.n_tracks = n_tracks
        self.n_measures = n_measures
        self.measure_resolution = measure_resolution
        self.n_pitches = n_pitches
        self.conditioning = conditioning
        self.conditioning_dim = conditioning_dim
        super().__init__()
        # pitch-time private
        self.pitch_time_p_0 = torch.nn.ModuleList([
            DiscriminatorBlock(1, 16, (1, 1, 12), (1, 1, 12)) for _ in range(n_tracks)
        ])
        self.pitch_time_p_1 = torch.nn.ModuleList([
            DiscriminatorBlock(16, 32, (1, 4, 1), (1, 4, 1)) for _ in range(n_tracks)
        ])

        # time-pitch private
        self.time_pitch_p_0 = torch.nn.ModuleList([
            DiscriminatorBlock(1, 16, (1, 4, 1), (1, 4, 1)) for _ in range(n_tracks)
        ])
        self.time_pitch_p_1 = torch.nn.ModuleList([
            DiscriminatorBlock(16, 32, (1, 1, 12), (1, 1, 12)) for _ in range(n_tracks)
        ])

        # merged private
        self.merged_p = torch.nn.ModuleList([
            DiscriminatorBlock(64, 64, (1, 1, 1), (1, 1, 1)) for _ in range(n_tracks)
        ])
        # h: 64 filters, 4 x 4 x 7

        # shared
        self.shared_0 = DiscriminatorBlock(64 * n_tracks, 128, (1, 2, 3), (1, 2, 1))
        self.shared_1 = DiscriminatorBlock(128, 256, (1, 2, 4), (1, 2, 4))

        # all merge
        self.all_merge0 = DiscriminatorBlock(256, 512, (2, 1, 1), (1, 1, 1))  # (3, 1, 1)
        self.all_merge1 = DiscriminatorBlock(512, 512, (3, 1, 1), (1, 1, 1))
        self.dense = torch.nn.Linear(512, 1)

    def forward(self, x):
        x = x.view(-1, self.n_tracks, self.n_measures, self.measure_resolution, self.n_pitches)
        pt_out = [conv(x[:, [i]]) for i, conv in enumerate(self.pitch_time_p_0)]
        pt_out = [conv(x_) for x_, conv in zip(pt_out, self.pitch_time_p_1)]
        tp_out = [conv(x[:, [i]]) for i, conv in enumerate(self.time_pitch_p_0)]
        tp_out = [conv(x_) for x_, conv in zip(tp_out, self.time_pitch_p_1)]

        h = [torch.cat([pt_out[i], tp_out[i]], 1) for i in range(self.n_tracks)]

        # merged private
        h = [conv(h[i]) for i, conv in enumerate(self.merged_p)]
        # h: [64 filters, 4 x 4 x 7], n_tracks

        h = torch.cat(h, 1)
        # h: 64 filters, 4 x 4 x 7

        # shared
        h = self.shared_0(h)
        h = self.shared_1(h)

        # all merge
        h = self.all_merge0(h)
        h = self.all_merge1(h)

        h = h.view(-1, 512)
        h = self.dense(h)
        return h


class Encoder(torch.nn.Module):
    """Convert piano-roll to latent expression.
    This model architecture is similar to Discriminator.
    """
    def __init__(self, n_tracks, n_measures, measure_resolution, n_pitches, output_dim):
        self.n_tracks = n_tracks
        self.n_measures = n_measures
        self.measure_resolution = measure_resolution
        self.n_pitches = n_pitches
        self.output_dim = output_dim
        super().__init__()
        self.conv0 = torch.nn.ModuleList([
            DiscriminatorBlock(1, 16, (1, 1, 12), (1, 1, 12)) for _ in range(n_tracks)
        ])
        self.conv1 = torch.nn.ModuleList([
            DiscriminatorBlock(16, 16, (1, 4, 1), (1, 4, 1)) for _ in range(n_tracks)
        ])
        self.conv2 = DiscriminatorBlock(16 * n_tracks, 64, (1, 1, 3), (1, 1, 1))
        self.conv3 = DiscriminatorBlock(64, 64, (1, 1, 4), (1, 1, 4))
        self.conv4 = DiscriminatorBlock(64, 128, (1, 4, 1), (1, 4, 1))
        self.conv5 = DiscriminatorBlock(128, 128, (2, 1, 1), (1, 1, 1))
        self.conv6 = DiscriminatorBlock(128, 256, (3, 1, 1), (3, 1, 1))
        self.dense = torch.nn.Linear(256, output_dim)

    def forward(self, x):
        x = x.view(-1, self.n_tracks, self.n_measures, self.measure_resolution, self.n_pitches)
        x = [conv(x[:, [i]]) for i, conv in enumerate(self.conv0)]
        x = torch.cat([conv(x_) for x_, conv in zip(x, self.conv1)], 1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1, 256)
        x = self.dense(x)
        return x
