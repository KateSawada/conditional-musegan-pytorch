from IPython.display import clear_output

import os
import os.path
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import pypianoroll
from pypianoroll import Multitrack, Track
from tqdm import tqdm
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot

from model import Discriminator, Encoder, Generator

def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

if __name__ == '__main__':
    # Data
    n_tracks = 5  # number of tracks
    n_pitches = 72  # number of pitches
    lowest_pitch = 24  # MIDI note number of the lowest pitch
    n_samples_per_song = 8  # number of samples to extract from each song in the datset
    n_measures = 4  # number of measures per sample
    beat_resolution = 4  # temporal resolution of a beat (in timestep)
    programs = [0, 0, 25, 33, 48]  # program number for each track
    is_drums = [True, False, False, False, False]  # drum indicator for each track
    track_names = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']  # name of each track
    tempo = 100

    # Training
    batch_size = 16
    latent_dim = 128
    n_steps = 20000

    # Sampling
    sample_interval = 100  # interval to run the sampler (in step)
    n_samples = 4

    measure_resolution = 4 * beat_resolution

    dataset_root = Path("data/lpd_5/lpd_5_cleansed/")
    id_list = []
    for path in os.listdir("data/amg"):
        filepath = os.path.join("data/amg", path)
        if os.path.isfile(filepath):
            with open(filepath) as f:
                id_list.extend([line.rstrip() for line in f])
    id_list = list(set(id_list))
    id_list = id_list[:32]

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

    data = torch.as_tensor(data, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(data)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    d = Discriminator(
        n_tracks=5,
        n_measures=4,
        measure_resolution=16,
        n_pitches=72,
        )
    e = Encoder(
        n_tracks=5,
        n_measures=4,
        measure_resolution=16,
        n_pitches=72,
        output_dim=64,
    )
    conditioning_dim = {
        "shared0": 256,
        "shared1": 128,
        "shared2": 64,
        "pt0": 32,
        "pt1": 16,
        "tp0": 32,
        "tp1": 16,
    }
    g = Generator(
        latent_dim=128,
        n_tracks=5,
        n_measures=4,
        measure_resolution=16,
        n_pitches=72,
        conditioning_dim=conditioning_dim,
        )
    for s in data_loader:
        c = e(s[0], True)
        for k in c.keys():
            print(k, end=": ")
            if isinstance(c[k], list):
                print(c[k][0].shape)
            else:
                print(c[k].shape)

        noise = torch.rand((16, 128))
        g_out = g(noise, c)
        d_out = d(g_out)
        print(d_out)
        print(d(s[0]))

        break
