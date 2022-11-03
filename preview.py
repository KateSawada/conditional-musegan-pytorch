import os

import torch
from torch.utils.tensorboard import SummaryWriter

from model import create_generator_from_config
from model import Discriminator

from custom import config
from custom import get_argument_parser


def preview(args, config):
    measure_resolution = 4 * config.beat_resolution
    generator = create_generator_from_config(config)

    discriminator = Discriminator(config.n_tracks, config.n_measures, measure_resolution, config.n_pitches)

    # create input tensor
    latent = torch.randn(config.batch_size, config.latent_dim)
    fake_samples = generator(latent)

    # write discriminator
    summary_writer = SummaryWriter(os.path.join(args.model_dir, "discriminator"))
    summary_writer.add_graph(discriminator, fake_samples.detach())
    summary_writer.close()

    # write generator
    summary_writer = SummaryWriter(os.path.join(args.model_dir, "generator"))
    summary_writer.add_graph(generator, latent)
    summary_writer.close()


if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    config.load(args.model_dir, args.configs, initialize=True)
    preview(args, config)
