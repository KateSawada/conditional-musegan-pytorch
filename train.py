import os
import os.path
import random
from pathlib import Path
import argparse
import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import pypianoroll
from pypianoroll import Multitrack, Track
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import create_generator_from_config
from model import Discriminator, Encoder, Generator
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
    # initialize loss metrics dict
    loss_dict = defaultdict(float)

    # Sample from the lantent distribution
    latent = torch.randn(batch_size, latent_dim)

    # Transfer data to GPU
    if is_cuda:
        real_samples = real_samples.cuda()
        latent = latent.cuda()

    conditioning_dim = encoder.output_dim
    condition = {}
    with torch.inference_mode():
        enc_out = encoder(real_samples, True)
        for k in enc_out.keys():
            if isinstance(enc_out[k], list):
                condition[k] = []
                for i in range(len(enc_out[k])):
                    condition[k].append(F.normalize(enc_out[k][i]))
            else:
                condition[k] = F.normalize(enc_out[k])

    # Generate fake samples with the generator
    fake_samples = generator(latent, condition)
    # binarize tensor input to discriminator as fake
    fake_sample_binarized = discretize(fake_samples.detach())

    # === Train the generator ===
    # pianoroll reconstruct loss
    if (config.g_pianoroll_reconstruct_loss == "BCE"):
        g_pianoroll_recon_loss_func = torch.nn.BCELoss()
    elif (config.g_pianoroll_reconstruct_loss == "L2"):
        g_pianoroll_recon_loss_func = torch.nn.MSELoss()
    elif (config.g_pianoroll_reconstruct_loss == "L1"):
        g_pianoroll_recon_loss_func = torch.nn.L1Loss()
    g_pianoroll_recon_loss = g_pianoroll_recon_loss_func(fake_samples, real_samples)
    loss_dict["g_pianoroll_recon_loss"] = g_pianoroll_recon_loss.item()
    # g_recon_loss.backward()

    # embedding reconstruct loss
    if (config.g_embedding_reconstruct_loss == "COS"):
        g_embedding_recon_loss_func = F.cosine_similarity
    # if (config.g_embedding_reconstruct_loss == "L2"):
    #     g_embedding_recon_loss_func = torch.nn.MSELoss()
    # elif (config.g_embedding_reconstruct_loss == "L1"):
    #     g_embedding_recon_loss_func = torch.nn.L1Loss()
    with torch.inference_mode():
        fake_samples_embedding = encoder(fake_samples)
    g_embedding_recon_loss = g_embedding_recon_loss_func(fake_samples_embedding, condition["shared0"]).mean()
    loss_dict["g_embedding_recon_loss"] = g_embedding_recon_loss.item()

    # Get discriminator outputs for the fake samples
    with torch.no_grad():
        prediction_fake_g = discriminator(fake_samples)
    # discriminator related loss
    if (config.loss == "mse"):
        g_adv_loss = F.mse_loss(prediction_fake_g, prediction_fake_g.new_ones(prediction_fake_g.size()))
    elif (config.loss == "hinge"):
        g_adv_loss = -prediction_fake_g.mean()
    loss_dict["g_adv_loss"] = g_adv_loss.item()

    # sum generator loss
    g_loss = g_pianoroll_recon_loss * config.g_pianoroll_reconstruct_loss_weight + g_embedding_recon_loss * config.g_embedding_reconstruct_loss_weight + g_adv_loss
    loss_dict["g_loss"] = g_loss.item()

    # Reset cached gradients to zero
    g_optimizer.zero_grad()
    # backward and grad_clip
    g_loss.backward()
    if config.generator_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(
            generator.parameters(),
            config.generator_grad_norm,
        )
    # Update the weights
    g_optimizer.step()

    # === Train the discriminator ===
    # Re-Generate fake samples with the generator
    with torch.no_grad():
        fake_samples = generator(latent, condition)

    prediction_fake_d = discriminator(fake_sample_binarized)
    prediction_real = discriminator(real_samples)
    # Compute the loss function
    if (config.loss == "mse"):
        d_loss_real = F.mse_loss(prediction_real, prediction_real.new_ones(prediction_real.size()))
        d_loss_fake = F.mse_loss(prediction_fake_d, prediction_fake_d.new_zeros(prediction_fake_d.size()))
    elif (config.loss == "hinge"):
        d_loss_real = -torch.mean(torch.min(prediction_real - 1, prediction_real.new_zeros(prediction_real.size())))
        d_loss_fake = -torch.mean(torch.min(-prediction_fake_d - 1, prediction_fake_d.new_zeros(prediction_fake_d.size())))
    loss_dict["d_loss_real"] = d_loss_real.item()
    loss_dict["d_loss_fake"] = d_loss_fake.item()

    # sum discriminator loss
    d_loss = d_loss_fake + d_loss_real
    loss_dict["d_loss"] = d_loss.item()

    # Reset cached gradients to zero
    d_optimizer.zero_grad()
    # Backpropagate the gradients
    d_loss.backward()
    if config.discriminator_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(),
                config.discriminator_grad_norm,
            )
    # Update the weights
    d_optimizer.step()

    return loss_dict

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
    conditioning_dim = config.conditioning_dim

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

    assert config.g_pianoroll_reconstruct_loss in ['BCE', 'L1', 'L2'], (
        "g_pianoroll_reconstruct_loss must be in ['BCE', 'L1', 'L2']"
    )
    assert config.g_embedding_reconstruct_loss in ['COS'], (
        "g_embedding_reconstruct_loss must be in ['COS']"
    )

    data_loader = gene_dataloader(config.train_json, batch=batch_size, shuffle=True)

    # Create neural networks
    discriminator = Discriminator(n_tracks, n_measures, measure_resolution, n_pitches)
    generator = Generator(
        latent_dim=latent_dim,
        n_tracks=n_tracks,
        n_measures=n_measures,
        measure_resolution=measure_resolution,
        n_pitches=n_pitches,
        conditioning_dim=conditioning_dim,
        )
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
    conditioning_model = config.conditioning_model  # TODO: triplet以外も追加?
    conditioning_model_pth = config.conditioning_model_pth
    conditioning_dim = config.conditioning_dim

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
            loss_dict = train_one_step(
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
                "(d_loss={: 8.6f}, g_loss={: 8.6f})".format(loss_dict["d_loss"], loss_dict["g_loss"]))

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

            # write to tensorboard
            for key, value in loss_dict.items():
                train_summary_writer.add_scalar(key, value, step)

            if (step % 10000 == 0):
                torch.save(generator.state_dict(), os.path.join(save_dir, f"generator-{step}.pth"))
                torch.save(discriminator.state_dict(), os.path.join(save_dir, f"discriminator-{step}.pth"))
            step += 1
            progress_bar.update(1)
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
