import random
from pathlib import Path
import glob

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pypianoroll
from pypianoroll import Multitrack, Track
from tqdm import tqdm
import os
import json


def fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


fix_seed()

def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)


def create_data(beat_resolution=4, lowest_pitch=24, n_pitches=72, n_measures=4):
    measure_resolution = 4 * beat_resolution

    target_dir = Path("./data/")

    musegan_root = "/home/ksawada/Documents/lab/lab_research/musegan-pytorch"
    dataset_root = os.path.join(musegan_root, "data/lpd_5/lpd_5_cleansed")

    id_list = []
    for path in os.listdir(os.path.join(musegan_root, "data/amg")):
        filepath = os.path.join(musegan_root, "data/amg", path)
        if os.path.isfile(filepath):
            with open(filepath) as f:
                id_list.extend([line.rstrip() for line in f])
    id_list = list(set(id_list))

    # Iterate over all the songs in the ID list
    for i, msd_id in enumerate(tqdm(id_list)):
        # Load the multitrack as a pypianoroll.Multitrack instance
        song_dir = Path(dataset_root) / msd_id_to_dirs(msd_id)
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

        # 先頭の空白の小節をカット
        measure_offset = 0
        while (pianoroll[:, measure_offset * n_measures * measure_resolution:(measure_offset + 1) * n_measures * measure_resolution, :].sum() == 0):
            measure_offset += 1
        pianoroll = pianoroll[:, measure_offset * n_measures * measure_resolution:, :]
        n_total_measures = pianoroll.shape[1] // measure_resolution

        target_n_samples = n_total_measures // n_measures
        candidate = n_total_measures - n_measures

        datadir = os.path.join(target_dir, f"{str(i).zfill(8)}_{msd_id}")
        os.makedirs(datadir, exist_ok=True)
        # Randomly select a number of phrases from the multitrack pianoroll
        for idx in range(candidate):
        # for idx in np.random.choice(candidate, target_n_samples, False):
            start = idx * measure_resolution
            end = (idx + n_measures) * measure_resolution
            # Skip the samples where some track(s) has too few notes
            if (pianoroll[:, start:end].sum(axis=(1, 2)) < 10).any():
                continue
            np.save(os.path.join(datadir, str(idx)), pianoroll[:, start:end])
    return


def load_data(id_list, dataset_root, beat_resolution, lowest_pitch,
              n_pitches, measure_resolution, n_measures, n_samples_per_song):
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

    np.save("data.npy", data)
    return data


def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    if max - min < 1e-5:
        return x
    else:
        result = (x - min) / ((max - min) + 1e-6)
    return result


class PiecesSet(Dataset):
    def __init__(self, song_dict):
        self.data = []
        self.labels = []
        self.song_dict = song_dict

        idx = 0
        for key in self.song_dict.keys():
            for segment in self.song_dict[key]:
                self.data.append(segment)
                self.labels.append(idx)
            idx += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        if os.path.exists(file_path):
            img = np.load(file_path)
            return (
                torch.from_numpy(img.astype(np.float32)).clone(),
                torch.from_numpy(
                    np.array(self.labels[idx]).astype(np.float32)
                ).clone(),
            )
        else:
            pass


def gene_dataloader(
    json_path, batch, shuffle=True
):
    op = open(json_path, "r")
    songs_dict = json.load(op)
    data = PiecesSet(songs_dict)
    loader = DataLoader(data, batch_size=batch, shuffle=shuffle, drop_last=True)

    return loader
