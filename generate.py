import datetime
import os
from operator import attrgetter
from typing import TYPE_CHECKING, Dict, Optional, Union
from copy import deepcopy

import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import torch
import pypianoroll
from pypianoroll import Multitrack, Track, StandardTrack, BinaryTrack
import pretty_midi
from pretty_midi import Instrument, PrettyMIDI

from model import create_generator_from_config
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
                velocity=int(velocity),
                pitch=pitch,
                start=note_on_times[idx],
                end=note_off_times[idx],
            )
            instrument.notes.append(note)

        instrument.notes.sort(key=attrgetter("start"))
        midi.instruments.append(instrument)

    return midi


def generate(args, config):
    """generate midi"""
    fix_seed(config.seed)
    tempo = config.tempo
    measure_resolution = 4 * config.beat_resolution


    tempo_array = np.full((4 * 4 * measure_resolution, 1), tempo)


    generator = create_generator_from_config(config)
    # generator.load_state_dict(torch.load("../exp/trial/results/checkpoint/generator-20000.pth"))
    generator.load_state_dict(torch.load(config.pth))

    # Prepare the inputs for the sampler, which wil run during the training
    sample_latent = torch.randn(config.n_samples, config.latent_dim)

    generator.eval()
    samples = generator(sample_latent).cpu().detach().numpy()

    # Display generated samples
    samples = samples.transpose(1, 0, 2, 3).reshape(config.n_tracks, -1, config.n_pitches)
    # print(samples.shape)  # (5, 256, 72)
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

    out_dir = os.path.join(config.out_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    os.makedirs(out_dir)
    config.save(out_dir)
    plt.savefig(os.path.join(out_dir, "pianoroll.png"))
    plt.show()

    to_pretty_midi(m).write(os.path.join(out_dir, "generated.mid"))
    # for i in range(config.n_tracks):
    #     pypianoroll.write(f"{i}.mid", tracks[i].standardize())
        # track = StandardTrack(program=config.program[i], is_drum=config.is_drum[i], pianoroll=samples[i])



if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    config.load(args.model_dir, args.configs, initialize=True)
    generate(args, config)
