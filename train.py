import os
import os.path
import random
from pathlib import Path
import argparse
import datetime

import numpy as np
import torch
import torch.nn.functional as F
import pypianoroll
from pypianoroll import Multitrack, Track
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import create_generator_from_config
from model import Discriminator, Encoder
from custom import config
from custom import get_argument_parser
from fix_seed import fix_seed
from dataload import gene_dataloader


def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

def load_data(id_list, dataset_root, beat_resolution, lowest_pitch,
              n_pitches, measure_resolution, n_measures, n_samples_per_song,
              filename):
    data = []
    # Iterate over all the songs in the ID list
    for msd_id in tqdm(id_list):
        # Load the multitrack as a pypianoroll.Multitrack instance
        song_dir = dataset_root / msd_id_to_dirs(msd_id)
        multitrack = pypianoroll.load(song_dir / os.listdir(song_dir)[0])
        # Binarize the pianorolls
        multitrack.binarize()
        # Downsample the pianorolls (shape: n_timesteps x n_pitches)
        multitrack.set_resolution(beat_resolution)
        # Stack the pianoroll (shape: n_tracks x n_timesteps x n_pitches)
        pianoroll = (multitrack.stack() > 0)
        # Get the target pitch range only
        pianoroll = pianoroll[:, :, lowest_pitch:lowest_pitch + n_pitches]
        # Calculate the total measures
        n_total_measures = multitrack.get_max_length() // measure_resolution
        candidate = n_total_measures - n_measures
        target_n_samples = min(n_total_measures // n_measures, n_samples_per_song)
        # Randomly select a number of phrases from the multitrack pianoroll
        for idx in np.random.choice(candidate, target_n_samples, False):
            start = idx * measure_resolution
            end = (idx + n_measures) * measure_resolution
            # Skip the samples where some track(s) has too few notes
            if (pianoroll.sum(axis=(1, 2)) < 10).any():
                continue
            data.append(pianoroll[:, start:end])
    # Stack all the collected pianoroll segments into one big array
    random.shuffle(data)
    data = np.stack(data)
    print(f"Successfully collect {len(data)} samples from {len(id_list)} songs")
    print(f"Data shape : {data.shape}")

    np.save(filename, data)
    return data

def compute_gradient_penalty(discriminator, real_samples, fake_samples, condition=None):
    """Compute the gradient penalty for regularization. Intuitively, the
    gradient penalty help stablize the magnitude of the gradients that the
    discriminator provides to the generator, and thus help stablize the training
    of the generator."""
    # Get random interpolations between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).cuda()
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
    interpolates = interpolates.requires_grad_(True)

    conditioning = (condition is not None)
    conditioning_dim = condition.shape[1]

    # Get the discriminator output for the interpolations
    d_interpolates = discriminator([interpolates, condition])
    # Get gradients w.r.t. the interpolations
    fake = torch.ones(real_samples.size(0), 1).cuda()
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    # Compute gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_one_step(d_optimizer, g_optimizer, real_samples,
                   generator, discriminator, batch_size, latent_dim, config,
                   is_cuda, encoder=None):
    """Train the networks for one step."""
    # Sample from the lantent distribution
    latent = torch.randn(batch_size, latent_dim)

    d_conditioning = config.discriminator_conditioning

    # Transfer data to GPU
    if is_cuda:
        real_samples = real_samples.cuda()
        latent = latent.cuda()

    g_conditioning = (encoder is not None)
    if g_conditioning:
        conditioning_dim = encoder.output_dim
        with torch.inference_mode():
            condition = encoder(real_samples)

    # === Train the discriminator ===
    # Reset cached gradients to zero
    d_optimizer.zero_grad()
    if (d_conditioning):
        # Get discriminator outputs for the real samples
        prediction_real = discriminator([real_samples, condition])
    else:
        prediction_real = discriminator(real_samples)

    # Compute the loss function
    # d_loss_real = torch.mean(torch.nn.functional.relu(1. - prediction_real))
    if (config.loss == "mse"):
        d_loss_real = F.mse_loss(prediction_real, prediction_real.new_ones(prediction_real.size()))
    elif (config.loss == "hinge"):
        d_loss_real = -torch.mean(torch.min(prediction_real - 1, prediction_real.new_zeros(prediction_real.size())))

    # Backpropagate the gradients
    d_loss_real.backward()

    if (g_conditioning):
        # Generate fake samples with the generator
        fake_samples = generator([latent, condition])
    else:
        fake_samples = generator(latent)
    # binarize tensor input to discriminator as fake
    fake_sample_binarized = discretize(fake_samples.detach())

    # Get discriminator outputs for the fake samples
    if (d_conditioning):
        prediction_fake_d = discriminator([fake_sample_binarized, condition])
    else:
        prediction_fake_d = discriminator(fake_sample_binarized)

    # Compute the loss function
    # d_loss_fake = torch.mean(torch.nn.functional.relu(1. + prediction_fake_d))
    if (config.loss == "mse"):
        d_loss_fake = F.mse_loss(prediction_fake_d, prediction_fake_d.new_zeros(prediction_fake_d.size()))
    elif (config.loss == "hinge"):
        d_loss_fake = -torch.mean(torch.min(-prediction_fake_d - 1, prediction_fake_d.new_zeros(prediction_fake_d.size())))

    # Backpropagate the gradients
    d_loss_fake.backward()

    # # Compute gradient penalty
    # gradient_penalty = 10.0 * compute_gradient_penalty(
    #     discriminator, real_samples.data, fake_samples.data,
    #     condition.data)
    # # Backpropagate the gradients
    # gradient_penalty.backward()

    if config.discriminator_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(),
                config.discriminator_grad_norm,
            )

    # Update the weights
    d_optimizer.step()

    # === Train the generator ===
    # Reset cached gradients to zero
    g_optimizer.zero_grad()
    # Get discriminator outputs for the fake samples
    if (d_conditioning):
        prediction_fake_g = discriminator([fake_samples, condition])
    else:
        prediction_fake_g = discriminator(fake_samples)
    # Compute the loss function
    if (config.loss == "mse"):
        g_loss = F.mse_loss(prediction_fake_g, prediction_fake_g.new_ones(prediction_fake_g.size()))
    elif (config.loss == "hinge"):
        g_loss = -prediction_fake_g.mean()

    # Backpropagate the gradients
    g_loss.backward(retain_graph=True)

    if (config.g_reconstruct_loss == "BCE"):
        g_recon_loss_func = torch.nn.BCELoss()
    elif (config.g_reconstruct_loss == "L2"):
        g_recon_loss_func = torch.nn.MSELoss()
    elif (config.g_reconstruct_loss == "L1"):
        g_recon_loss_func = torch.nn.L1Loss()
    g_recon_loss = g_recon_loss_func(fake_samples, real_samples) * config.g_reconstruct_loss_weight
    g_recon_loss.backward()

    if config.generator_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                generator.parameters(),
                config.generator_grad_norm,
            )

    # Update the weights
    g_optimizer.step()

    return d_loss_real + d_loss_fake, g_loss, g_recon_loss

def discretize(x, threshold=0.5):
    """discretize Tensor to 0/1 by thresholding.

    Args:
        x (torch.Tensor): tensor
        threshold (int or float): threshold. default to 0.5

    Returns:
        torch.Tensor
    """
    threshold = torch.Tensor([threshold])
    if x.is_cuda:
        threshold = threshold.cuda()
    discretized_x = (x > threshold).float()
    return discretized_x

def train(args, config):
    fix_seed(config.seed)

    # Data
    n_tracks = config.n_tracks  # number of tracks
    n_pitches = config.n_pitches  # number of pitches
    lowest_pitch = config.lowest_pitch  # MIDI note number of the lowest pitch
    n_samples_per_song = config.n_samples_per_song  # number of samples to extract from each song in the dataset
    n_measures = config.n_measures  # number of measures per sample
    beat_resolution = config.beat_resolution  # temporal resolution of a beat (in timestep)
    programs = config.programs  # program number for each track
    is_drums = config.is_drums  # drum indicator for each track
    track_names = config.track_names  # name of each track
    tempo = config.tempo
    latent_dim = config.latent_dim

    # Training
    batch_size = config.batch_size
    n_steps = config.n_steps

    measure_resolution = 4 * beat_resolution
    assert 24 % beat_resolution == 0, (
        "beat_resolution must be a factor of 24 (the beat resolution used in "
        "the source dataset)."
    )
    assert len(programs) == len(is_drums) and len(programs) == len(track_names), (
        "Lengths of programs, is_drums and track_names must be the same."
    )

    assert config.loss in ["mse", "hinge"], (
        "loss must be in ['mse', 'hinge']"
    )

    assert config.g_reconstruct_loss in ['BCE', 'L1', 'L2'], (
        "g_reconstruct_loss must be in ['BCE', 'L1', 'L2']"
    )

    data_loader = gene_dataloader(config.train_json, batch=batch_size, shuffle=True)

    # Create neural networks
    discriminator = Discriminator(n_tracks, n_measures, measure_resolution, n_pitches, config.discriminator_conditioning, config.conditioning_dim)
    generator = create_generator_from_config(config)
    print("Number of parameters in G: {}".format(
        sum(p.numel() for p in generator.parameters() if p.requires_grad)))
    print("Number of parameters in D: {}".format(
        sum(p.numel() for p in discriminator.parameters() if p.requires_grad)))

    # Create optimizers
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=0.001,  betas=(0.5, 0.9))
    g_optimizer = torch.optim.Adam(
        generator.parameters(), lr=0.001, betas=(0.5, 0.9))

    # conditioning config
    conditioning = config.conditioning
    conditioning_model = config.conditioning_model  # TODO: triplet????????????????
    conditioning_model_pth = config.conditioning_model_pth
    conditioning_dim = config.conditioning_dim

    if conditioning:
        encoder = Encoder(
            n_tracks, n_measures, measure_resolution, n_pitches,
            conditioning_dim)
        encoder.load_state_dict(torch.load(conditioning_model_pth))

    # Transfer the neural nets and samples to GPU
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        discriminator = discriminator.cuda()
        discriminator = torch.nn.DataParallel(discriminator)
        generator = generator.cuda()
        generator = torch.nn.DataParallel(generator)
        if conditioning:
            encoder = encoder.cuda()
    if (config.trained_g_model is not None):
        generator.load_state_dict(torch.load(config.trained_g_model))
        print(f"generator weights loaded from {config.trained_g_model}")
    if (config.trained_d_model is not None):
        discriminator.load_state_dict(torch.load(config.trained_d_model))
        print(f"discriminator weights loaded from {config.trained_d_model}")

    # Create an empty dictionary to sotre history samples
    # history_samples = {}

    # Create a LiveLoss logger instance for monitoring
    # liveloss = PlotLosses(outputs=[MatplotlibPlot(cell_size=(6,2))])

    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    save_dir = os.path.join(args.model_dir, f"{config.experiment}-{current_time}")
    os.makedirs(save_dir)
    config.save(save_dir)
    train_log_dir = os.path.join(save_dir, "train")
    # eval_log_dir = os.path.join(save_dir, "eval")
    train_summary_writer = SummaryWriter(train_log_dir)
    # train_summary_writer.add_graph(generator, torch.randn(batch_size, latent_dim))
    # train_summary_writer.add_graph(discriminator, next(iter(data_loader))[0])
    # eval_summary_writer = SummaryWriter(eval_log_dir)


    # Initialize step
    step = 0

    # Create a progress bar instance for monitoring
    progress_bar = tqdm(total=n_steps, initial=step, ncols=80, mininterval=1)

    # Start iterations
    while step < n_steps + 1:
        # Iterate over the dataset
        for real_samples in data_loader:
            # Train the neural networks
            generator.train()
            d_loss, g_loss, g_recon_loss = train_one_step(
                d_optimizer, g_optimizer, real_samples[0],
                generator, discriminator, batch_size, latent_dim, config,
                is_cuda, encoder)

            # Record smoothened loss values to LiveLoss logger
            # if step > 0:
            #     running_d_loss = 0.05 * d_loss + 0.95 * running_d_loss
            #     running_g_loss = 0.05 * g_loss + 0.95 * running_g_loss
            # else:
            #     running_d_loss, running_g_loss = 0.0, 0.0
            # liveloss.update({'negative_critic_loss': -running_d_loss})
            # liveloss.update({'d_loss': running_d_loss, 'g_loss': running_g_loss})

            # Update losses to progress bar
            progress_bar.set_description_str(
                "(d_loss={: 8.6f}, g_loss={: 8.6f})".format(d_loss, g_loss))

            #
            # if step % sample_interval == 0:
            #     # Get generated samples
            #     generator.eval()
            #     samples = generator(sample_latent).cpu().detach().numpy()
            #     history_samples[step] = samples

            #     # Display loss curves
            #     # clear_output(True)
            #     # if step > 0:
            #     #     liveloss.send()

            #     # Display generated samples
            #     samples = samples.transpose(1, 0, 2, 3).reshape(n_tracks, -1, n_pitches)
            #     tracks = []
            #     for idx, (program, is_drum, track_name) in enumerate(
            #         zip(programs, is_drums, track_names)
            #     ):
            #         pianoroll = np.pad(
            #             samples[idx] > 0.5,
            #             ((0, 0), (lowest_pitch, 128 - lowest_pitch - n_pitches))
            #         )
            #         tracks.append(
            #             Track(
            #                 name=track_name,
            #                 program=program,
            #                 is_drum=is_drum,
            #                 pianoroll=pianoroll
            #             )
            #         )
            #
            train_summary_writer.add_scalar("g_loss", g_loss, step)
            train_summary_writer.add_scalar("d_loss", d_loss, step)
            train_summary_writer.add_scalar("g_recon_loss", g_recon_loss, step)

            if (step % 10000 == 0):
                torch.save(generator.state_dict(), os.path.join(save_dir, f"generator-{step}.pth"))
                torch.save(discriminator.state_dict(), os.path.join(save_dir, f"discriminator-{step}.pth"))
            step += 1
            progress_bar.update(1)
            del d_loss, g_loss, g_recon_loss
            if step >= n_steps:
                break
    train_summary_writer.close()
    torch.save(generator.state_dict(), os.path.join(args.model_dir, "generator-final.pth"))
    torch.save(discriminator.state_dict(), os.path.join(args.model_dir, "discriminator-final.pth"))


if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    config.load(args.model_dir, args.configs, initialize=True)
    train(args, config)
