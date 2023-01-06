import argparse
import json
import os

import yaml
import pretty_midi
import numpy as np
from pypianoroll import Multitrack, Track
import pypianoroll

from generate import to_pretty_midi, midi_to_wav


def np_to_midi(ary: np.ndarray, config:dict) -> pretty_midi.PrettyMIDI:
    measure_resolution = 4 * config["beat_resolution"]
    tempo_array = np.full((4 * measure_resolution, 1), config["tempo"])
    tracks = []
    for idx, (program, is_drum, track_name) in enumerate(
        zip(config["programs"], config["is_drums"], config["track_names"])
    ):
        pianoroll = np.pad(
            ary[idx] > 0.5,
            ((0, 0), (config["lowest_pitch"], 128 - config["lowest_pitch"] - config["n_pitches"]))
        )
        tracks.append(
            Track(
                name=track_name,
                program=program,
                is_drum=is_drum,
                pianoroll=pianoroll
            )
        )
    m = Multitrack(tracks=tracks, tempo=tempo_array, resolution=config["beat_resolution"])
    mid = to_pretty_midi(m)
    return mid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True,
                        help="path to json file")
    parser.add_argument("-o", "--out", type=str, required=True,
                        help="path to output files directory")
    parser.add_argument("-c", "--config", default=[], required=True, nargs="*",
                        help="path to configuration yml file")
    args = parser.parse_args()

    # 出力先ディレクトリを作成
    if (not os.path.exists(args.out)):
        os.makedirs(args.out)

    config = {}
    # ymlを読み込み
    for c in args.config:
        yml = yaml.load(open(c).read(), Loader=yaml.FullLoader)
        for k, v in yml.items():
            config[k] = v

    # jsonを読み込み
    with open(args.file) as f:
        segments_lst = json.load(f)

    for segment_file in segments_lst:
        ary = np.load(segment_file)
        print(ary.shape)

        # ファイル名を整形
        filename = segment_file.replace("/", "_")
        filename = filename[1:] if filename[0] == "." else filename
        filename = filename[1:] if filename[0] == "_" else filename
        filename = filename.replace(".npy", "")
        print(filename)

        filepath = os.path.join(args.out, filename)

        midi = np_to_midi(ary, config)
        midi.write(filepath+".mid")
        midi_to_wav(filepath + ".mid", filepath + ".wav")


if __name__ == "__main__":
    main()
