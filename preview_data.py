from operator import attrgetter
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Optional, Union

import numpy as np
import pypianoroll
from pypianoroll import Multitrack, Track, StandardTrack, BinaryTrack
import pretty_midi
from pretty_midi import Instrument, PrettyMIDI
import matplotlib.pyplot as plt
import scipy.stats

from custom import config
from custom import get_argument_parser


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

if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    config.load(args.model_dir, args.configs, initialize=True)
    filename = "data/lpd_5/lpd_5_cleansed/A/A/A/TRAAAZF12903CCCF6B/05f21994c71a5f881e64f45c8d706165.npz"

    measure_resolution = 4 * config.beat_resolution

    multitrack = pypianoroll.load(filename)

    multitrack.binarize()
    multitrack.set_resolution(config.beat_resolution)
    samples = (multitrack.stack() > 0)
    samples = samples[:, :, config.lowest_pitch:config.lowest_pitch + config.n_pitches]
    tempo_array = np.full((4 * 4 * measure_resolution, 1), config.tempo)

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

    # plt.savefig(os.path.join(out_dir, "pianoroll.png"))
    plt.show()

    to_pretty_midi(m).write("sample.mid")
