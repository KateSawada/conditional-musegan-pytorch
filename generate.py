import datetime
import os
from operator import attrgetter
from typing import TYPE_CHECKING, Dict, Optional, Union
from copy import deepcopy
import json
import random
import subprocess

import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pypianoroll
from pypianoroll import Multitrack, Track, StandardTrack, BinaryTrack
import pretty_midi
from pretty_midi import Instrument, PrettyMIDI

from model import create_generator_from_config, Encoder
from custom import config
from custom import get_argument_parser
from fix_seed import fix_seed


DEFAULT_TEMPO = 120


def to_pretty_midi(
    multitrack: "Multitrack",
    default_tempo: Optional[float] = None,
    default_velocity: int = 64,
) -> PrettyMIDI:
    """Return a Multitrack object as a PrettyMIDI object.

    Parameters
    ----------
    default_tempo : int
        Default tempo to use. Defaults to the first element of
        attribute `tempo`.
    default_velocity : int
        Default velocity to assign to binarized tracks. Defaults to
        64.

    Returns
    -------
    :class:`pretty_midi.PrettyMIDI`
        Converted PrettyMIDI object.

    Notes
    -----
    - Tempo changes are not supported.
    - Time signature changes are not supported.
    - The velocities of the converted piano rolls will be clipped to
      [0, 127].
    - Adjacent nonzero values of the same pitch will be considered
      a single note with their mean as its velocity.

    """
    if default_tempo is not None:
        tempo = default_tempo
    elif multitrack.tempo is not None:
        tempo = float(scipy.stats.hmean(multitrack.tempo))
    else:
        tempo = DEFAULT_TEMPO

    # Create a PrettyMIDI instance
    midi = PrettyMIDI(initial_tempo=tempo)

    # Compute length of a time step
    time_step_length = 60.0 / tempo / multitrack.resolution

    for track in multitrack.tracks:
        instrument = Instrument(
            program=track.program, is_drum=track.is_drum, name=track.name
        )
        track = track.standardize()
        if isinstance(track, BinaryTrack):
            processed = track.set_nonzeros(default_velocity)
        elif isinstance(track, StandardTrack):
            copied = deepcopy(track)
            processed = copied.clip()
        else:
            raise ValueError(
                f"Expect BinaryTrack or StandardTrack, but got {type(track)}."
            )
        clipped = processed.pianoroll.astype(np.uint8)
        binarized = clipped > 0
        padded = np.pad(binarized, ((1, 1), (0, 0)), "constant")
        diff = np.diff(padded.astype(np.int8), axis=0)

        positives = np.nonzero((diff > 0).T)
        pitches = positives[0]
        note_ons = positives[1]
        note_on_times = time_step_length * note_ons
        note_offs = np.nonzero((diff < 0).T)[1]
        note_off_times = time_step_length * note_offs

        for idx, pitch in enumerate(pitches):
            velocity = np.mean(clipped[note_ons[idx] : note_offs[idx], pitch])
            note = pretty_midi.Note(
                velocity=80,
                pitch=pitch,
                start=note_on_times[idx],
                end=note_off_times[idx],
            )
            instrument.notes.append(note)

        instrument.notes.sort(key=attrgetter("start"))
        midi.instruments.append(instrument)

    return midi


def midi_to_wav(midi_path:str, wav_path:str):
    subprocess.run(["timidity", midi_path, "-Ow", wav_path])


def generate(args, config):
    """generate midi"""
    fix_seed(config.seed)
    tempo = config.tempo
    measure_resolution = 4 * config.beat_resolution

    tempo_array = np.full((4 * 4 * measure_resolution, 1), tempo)

    if (config.generate_json):
        with open(config.generate_json, "r") as f:
            segments_list = json.load(f)
        # print(segments_list)
        # print(type(segments_list))

    encoder = Encoder(config.n_tracks, config.n_measures, measure_resolution, config.n_pitches,
        config.conditioning_dim)
    encoder.load_state_dict(torch.load(config.conditioning_model_pth))

    generator = create_generator_from_config(config)
    generator = torch.nn.DataParallel(generator)
    # generator.load_state_dict(torch.load("../exp/trial/results/checkpoint/generator-20000.pth"))
    generator.load_state_dict(torch.load(config.pth))
    generator.eval()

    for song in segments_list.keys():
        out_dir = os.path.join(config.out_dir, os.path.basename(song))
        os.makedirs(out_dir)
        os.makedirs(os.path.join(out_dir, "pianoroll"))
        os.makedirs(os.path.join(out_dir, "wav"))
        os.makedirs(os.path.join(out_dir, "npy"))
        os.makedirs(os.path.join(out_dir, "mid"))


        for seg in segments_list[song]:
            segment_name = os.path.splitext(os.path.basename(seg))[0]
            conditions = F.normalize(encoder(torch.from_numpy(np.load(seg).astype(np.float32))))
            if ("add_noise" in config.dict.keys() and config.add_noise):
                conditions += torch.normal(mean=0, std=config.additional_noise_std, size=conditions.shape)

            sample_latent = torch.randn(1, config.latent_dim)
            samples = generator([sample_latent, conditions]).cpu().detach().numpy()
            samples = samples.transpose(1, 0, 2, 3).reshape(config.n_tracks, -1, config.n_pitches)

            if ("n_repeat" in config.dict.keys()):
                for _ in range(int(config.n_repeat) - 1):
                    sample_latent = torch.randn(1, config.latent_dim)
                    samples_ = generator([sample_latent, conditions]).cpu().detach().numpy()
                    samples_ = samples_.transpose(1, 0, 2, 3).reshape(config.n_tracks, -1, config.n_pitches)

                    samples = np.concatenate((samples, samples_), axis=1)


            # Display generated samples


            tracks = []
            for idx, (program, is_drum, track_name) in enumerate(
                zip(config.programs, config.is_drums, config.track_names)
            ):
                pianoroll = np.pad(
                    samples[idx] > 0.5,
                    ((0, 0), (config.lowest_pitch, 128 - config.lowest_pitch - config.n_pitches))
                )
                tracks.append(
                    Track(
                        name=track_name,
                        program=program,
                        is_drum=is_drum,
                        pianoroll=pianoroll
                    )
                )
            m = Multitrack(tracks=tracks, tempo=tempo_array, resolution=config.beat_resolution)

            axs = m.plot()

            for ax in axs:
                for x in range(
                    measure_resolution,
                    4 * measure_resolution * config.n_measures,
                    measure_resolution
                ):
                    if x % (measure_resolution * 4) == 0:
                        ax.axvline(x - 0.5, color='k')
                    else:
                        ax.axvline(x - 0.5, color='k', linestyle='-', linewidth=1)
            plt.gcf().set_size_inches((16, 8))

            plt.savefig(os.path.join(out_dir, "pianoroll", f"{segment_name}.png"))
            # plt.show()

            to_pretty_midi(m).write(os.path.join(out_dir, "mid", f"{segment_name}.mid"))
            # for i in range(config.n_tracks):
            #     pypianoroll.write(f"{i}.mid", tracks[i].standardize())
                # track = StandardTrack(program=config.program[i], is_drum=config.is_drum[i], pianoroll=samples[i])
            midi_to_wav(os.path.join(out_dir, "mid", f"{segment_name}.mid"), os.path.join(out_dir, "wav", f"{segment_name}.wav"))
            np.save(os.path.join(out_dir, "npy", f"{segment_name}.npy"), samples)

if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    config.load(args.model_dir, args.configs, initialize=True)
    generate(args, config)
