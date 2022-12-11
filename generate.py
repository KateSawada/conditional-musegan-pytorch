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
    generator = torch.nn.DataParallel(generator)
    # generator.load_state_dict(torch.load("../exp/trial/results/checkpoint/generator-20000.pth"))
    generator.load_state_dict(torch.load(config.pth))

    # Prepare the inputs for the sampler, which wil run during the training
    sample_latent = torch.randn(config.n_samples, config.latent_dim)

    # [0]
    condition = np.array([-0.02841583, -0.08944083, -0.13972868, -0.00338366,  0.1702567 ,
       -0.1861044 ,  0.18723449,  0.0672648 , -0.17370182,  0.07689918,
        0.08475344, -0.08417809,  0.13771936,  0.14094101,  0.2338504 ,
       -0.18342003,  0.09372227, -0.13299803, -0.11893179, -0.20410345,
       -0.19835322,  0.07111041, -0.06870961, -0.04113474,  0.14238904,
        0.00795381, -0.00587721,  0.12777096,  0.05284411, -0.2972005 ,
        0.19495434,  0.11463467, -0.04622035, -0.01966416, -0.02742159,
        0.09943493,  0.07251707, -0.13324353, -0.19073258,  0.05377371,
       -0.00565432,  0.21793903, -0.04262271, -0.06480663,  0.09376774,
        0.13846368,  0.06013491,  0.0237317 , -0.16026707,  0.02772263,
       -0.17822324, -0.13624203,  0.14403662,  0.02924404,  0.1172649 ,
       -0.02817015,  0.0356379 ,  0.12646827, -0.09503642, -0.0904337 ,
        0.14770986, -0.02157568,  0.15055102, -0.15936786])

    # 5000
    condition = np.array([ 0.17578349,  0.04186243, -0.12407853,  0.04051666,  0.01025327,
        0.1556667 , -0.0896385 ,  0.09224521,  0.05592323,  0.01038782,
       -0.12069879, -0.01249089, -0.05347871, -0.03622057, -0.17468464,
        0.05432663,  0.01074304,  0.00833865,  0.07926927,  0.03885436,
        0.07410286,  0.24883522,  0.05017756,  0.01659776, -0.12030864,
       -0.12579861, -0.15648594, -0.0549384 , -0.05590904, -0.01357666,
       -0.04918401, -0.2105002 ,  0.11821821, -0.06766159, -0.15812498,
       -0.25550982,  0.18167123,  0.15016738,  0.07314271, -0.10344204,
       -0.23316382, -0.0511257 ,  0.11413497,  0.06687328,  0.15769508,
        0.07387913, -0.09642395,  0.00660014,  0.07377002,  0.01917652,
       -0.1243469 ,  0.04781255, -0.17110102,  0.09839039, -0.22903807,
        0.06300498,  0.08409222, -0.06287495,  0.04869368,  0.40754077,
       -0.01235847, -0.05245803, -0.26334327, -0.05030605])

    # 10000
    condition = np.array([ 0.01856193, -0.01477286,  0.03468726,  0.10118691, -0.20805198,
       -0.05546994, -0.06952308, -0.14709963, -0.10149267,  0.16011786,
       -0.17491096, -0.0034203 , -0.03415287,  0.17581014,  0.03884406,
       -0.0567137 ,  0.05969622, -0.24289371, -0.15543304,  0.24937479,
        0.21932636,  0.17022513, -0.00180162,  0.293221  ,  0.08445243,
        0.12189536,  0.03126807,  0.13299832,  0.04373781,  0.003554  ,
       -0.1355431 , -0.15922284, -0.04554602, -0.09243896, -0.0586375 ,
        0.09779994, -0.0962754 , -0.09624911,  0.11489519,  0.08208706,
        0.02079623,  0.11245488, -0.05925377,  0.07507263,  0.24386573,
        0.07467891, -0.12431281,  0.14556184, -0.08810599,  0.01810817,
        0.12589036, -0.11222643, -0.02040146,  0.15126833,  0.08499465,
       -0.03154435, -0.11126968,  0.00402572, -0.06862617, -0.07703833,
       -0.2742678 , -0.20822518,  0.09158227, -0.09730782])

    # 100000
    condition = np.array([-0.1219351 ,  0.14558016, -0.01255446,  0.07696897, -0.03343865,
       -0.139151  ,  0.10247305, -0.21507694, -0.02300637, -0.12605666,
        0.02114416, -0.03266053,  0.07401488, -0.14195108,  0.14796486,
        0.01705191,  0.06830101,  0.10592607, -0.102909  ,  0.04182998,
        0.07691199,  0.00467995,  0.14733103, -0.18301712,  0.10311784,
        0.08364564,  0.04893938, -0.00442747,  0.12172122, -0.08195998,
        0.11419528, -0.03190966, -0.15920435, -0.37607118, -0.07713049,
       -0.01877548, -0.04688122,  0.22673307,  0.10205754, -0.0711036 ,
        0.15517086, -0.18224843, -0.16015318, -0.04710748,  0.00265257,
        0.12145255,  0.22897598,  0.1327008 , -0.21134742,  0.00932523,
       -0.00456264, -0.10216224,  0.13078363,  0.02037127,  0.10617673,
       -0.08434818,  0.23833105,  0.14183933,  0.00225731, -0.05035898,
       -0.24570711,  0.12389705, -0.09114699,  0.0560828 ])

    # 100001
    condition = np.array([-1.19967930e-01,  9.78232548e-02, -4.72434759e-02,  9.80119780e-02,
       -4.52863611e-02, -1.99931577e-01,  7.47196749e-02, -2.09841505e-01,
        7.52611738e-03, -1.72499076e-01,  1.79808065e-02, -2.57309116e-02,
        1.05784625e-01, -7.28659928e-02,  1.40163884e-01, -4.49708011e-03,
        1.12249173e-01,  1.56815901e-01, -7.10383803e-02,  2.22308077e-02,
        7.54966913e-03, -2.09103487e-02,  1.24232210e-01, -1.51210114e-01,
        9.54637453e-02,  4.61637750e-02,  3.25271264e-02,  8.55230028e-05,
        1.39842808e-01, -1.13215081e-01,  1.74452454e-01, -6.61443733e-03,
       -1.47259861e-01, -3.75773817e-01, -6.43232763e-02, -1.84836239e-02,
       -7.77386129e-02,  1.85288012e-01,  9.18588638e-02, -8.54830295e-02,
        1.77896008e-01, -1.10443786e-01, -2.14670539e-01, -9.74854156e-02,
       -1.40865352e-02,  1.39781058e-01,  1.90000236e-01,  1.35655239e-01,
       -2.19291359e-01, -2.58351024e-03,  9.83980834e-04, -1.15954138e-01,
        1.61738619e-01,  1.82165708e-02,  6.90506697e-02, -7.12639019e-02,
        2.64258057e-01,  1.58801347e-01,  2.73225047e-02, -7.26215243e-02,
       -2.00215831e-01,  7.47617185e-02, -5.96427657e-02,  5.97766638e-02])

    # 150000
    # condition = np.array([ 0.03258145, -0.2338179 , -0.05869518, -0.03129718,  0.10297893,
    #     0.1001856 , -0.02068575, -0.01280872, -0.27825415,  0.14055184,
    #     0.03389766,  0.06991973,  0.04117396,  0.29669073,  0.07554618,
    #     0.07475223,  0.02094822, -0.07460605, -0.09004921, -0.08853813,
    #    -0.10769971, -0.07681512, -0.00894501, -0.25442272,  0.12940386,
    #     0.01363598,  0.16330256, -0.0374665 ,  0.14395353, -0.11089132,
    #    -0.02498437, -0.07683881, -0.09572726,  0.0619818 ,  0.07077526,
    #    -0.18043073,  0.33119345,  0.19877826, -0.04707282,  0.03157506,
    #    -0.00890703,  0.03484868,  0.28723133,  0.06702067, -0.05675841,
    #     0.00076744,  0.00180478, -0.11553767, -0.05100165,  0.06718601,
    #    -0.24282795,  0.03712592, -0.08385053,  0.00071755, -0.13223907,
    #    -0.05081627,  0.05079518,  0.13824257, -0.07886582,  0.00689047,
    #    -0.03777171, -0.20708728, -0.12324633,  0.18893158])

    condition = torch.from_numpy(condition.astype(np.float32)).clone()

    generator.eval()
    samples = generator([sample_latent, condition]).cpu().detach().numpy()

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
