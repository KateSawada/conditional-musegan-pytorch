import json
import glob
import os

import torch
import numpy as np

from model import Encoder


def empty_bars(tensor, n_measures=4, measure_resolution=16):
    """ratio of empty bars
    tensor.shape = [n_tracks, n_timestep, n_pitch]

    returns:
        ndarray(shape=(n_songs, n_tracks))
    """
    if (isinstance(tensor, torch.Tensor)):
        tensor = tensor.to('cpu').detach().numpy().copy()
    # 閾値処理
    tensor = np.where(tensor >= 0.5, 1, 0)
    song_resolution = n_measures * measure_resolution
    n_songs = tensor.shape[1] // (song_resolution)
    n_tracks = tensor.shape[0]
    empty_bars_ratio = np.zeros((n_songs, n_tracks))

    # 計算
    for i_song in range(n_songs):
        tensor_ = tensor[:, song_resolution * i_song : song_resolution * (i_song + 1)]
        for i_bar in range(n_measures):
            empty_bars_ratio[i_song] += np.all(tensor_[:, measure_resolution * i_bar : measure_resolution * (i_bar + 1)] == 0, axis=(1, 2))
    return empty_bars_ratio / n_measures


def used_pitch_classes(tensor, n_measures=4, measure_resolution=16):
    """number of used pitch classes per bar (from 0 to 12)
    tensor.shape = [n_tracks, n_timestep, n_pitch]
    """
    if (isinstance(tensor, torch.Tensor)):
        tensor = tensor.to('cpu').detach().numpy().copy()
    # 閾値処理
    tensor = np.where(tensor >= 0.5, 1, 0)

    song_resolution = n_measures * measure_resolution
    n_songs = tensor.shape[1] // (song_resolution)
    n_tracks = tensor.shape[0]

    upc = np.zeros((n_songs, n_tracks))

    # pitch方向の端数処理
    rem = tensor.shape[2] % 12
    if (rem != 0):
        reminder = tensor[:, :, -rem:]
        tensor = tensor[:, :, :-rem]
    tensor = np.reshape(tensor, (tensor.shape[0], tensor.shape[1], 12, -1))
    if (rem != 0):
        tensor[:, :, :rem] += reminder

    for i_song in range(n_songs):
        tensor_ = tensor[:, song_resolution * i_song : song_resolution * (i_song + 1)]
        for i_bar in range(n_measures):
            upc[i_song] += np.count_nonzero(np.sum(tensor_[:, measure_resolution * i_bar : measure_resolution * (i_bar + 1)], axis=(1, 3)), axis=1)
    return upc / n_measures


def drum_pattern(tensor, n_measures=4, measure_resolution=16, drum_track=0, tolerance=0.1):
    """number of used pitch classes per bar (from 0 to 12)
    tensor.shape = [n_tracks, n_timestep, n_pitch]
    """
    if (isinstance(tensor, torch.Tensor)):
        tensor = tensor.to('cpu').detach().numpy().copy()

    # 閾値処理
    tensor = np.where(tensor >= 0.5, 1, 0)

    song_resolution = n_measures * measure_resolution
    n_songs = tensor.shape[1] // (song_resolution)

    dp = np.zeros(n_songs)

    # ドラムトラック抽出
    tensor = tensor[drum_track]

    mask = np.tile((1, tolerance, 0, tolerance), 4 * n_measures)

    for i_song in range(n_songs):
        tensor_ = tensor[song_resolution * i_song : song_resolution * (i_song + 1)]

        dp[i_song] += np.sum(np.sum(tensor_, axis=1) * mask) / np.sum(tensor_)
    return dp / n_measures

def _tonal_matrix(r1=1.0, r2=1.0, r3=0.5):
    """Compute and return a tonal matrix for computing the tonal distance
    [1]. Default argument values are set as suggested by the paper.

    [1] Christopher Harte, Mark Sandler, and Martin Gasser. Detecting
    harmonic change in musical audio. In Proc. ACM MM Workshop on Audio and
    Music Computing Multimedia, 2006.
    """
    tonal_matrix = np.empty((6, 12))
    tonal_matrix[0] = r1 * np.sin(np.arange(12) * (7. / 6.) * np.pi)
    tonal_matrix[1] = r1 * np.cos(np.arange(12) * (7. / 6.) * np.pi)
    tonal_matrix[2] = r2 * np.sin(np.arange(12) * (3. / 2.) * np.pi)
    tonal_matrix[3] = r2 * np.cos(np.arange(12) * (3. / 2.) * np.pi)
    tonal_matrix[4] = r3 * np.sin(np.arange(12) * (2. / 3.) * np.pi)
    tonal_matrix[5] = r3 * np.cos(np.arange(12) * (2. / 3.) * np.pi)
    return tonal_matrix

def tonal_distance(tensor, n_measures=4, measure_resolution=16, drum_track=0):
    """number of used pitch classes per bar (from 0 to 12)
    tensor.shape = [n_tracks, n_timestep, n_pitch]
    """
    if (isinstance(tensor, torch.Tensor)):
        tensor = tensor.to('cpu').detach().numpy().copy()
    # 閾値処理
    tensor = np.where(tensor >= 0.5, 1, 0)

    song_resolution = n_measures * measure_resolution
    n_songs = tensor.shape[1] // (song_resolution)
    n_tracks = tensor.shape[0]

    tonal_matrix = _tonal_matrix()

    td = np.zeros((n_songs, n_tracks - 1, n_tracks - 1))

    # pitch方向の端数処理
    rem = tensor.shape[2] % 12
    if (rem != 0):
        reminder = tensor[:, :, -rem:]
        tensor = tensor[:, :, :-rem]
    tensor = np.reshape(tensor, (tensor.shape[0], tensor.shape[1], 12, -1))
    if (rem != 0):
        tensor[:, :, :rem] += reminder

    tensor = np.sum(tensor, axis=3)
    tensor = np.concatenate((tensor[: drum_track], tensor[drum_track + 1: ]), axis=0)

    tensor = np.reshape(tensor, (n_songs, -1, measure_resolution // 4, 12, n_tracks - 1))
    tensor = np.sum(tensor, axis=2)

    tensor = tensor / np.sum(tensor, axis=2, keepdims=True)
    tensor[np.isnan(tensor)] = 0  # 空白の小節ではzero divisionのためnanが出るため0に置換

    tensor = np.reshape(np.transpose(tensor, (0, 2, 1, 3)), (n_songs, n_tracks - 1, 12, -1))

    for i_song in range(n_songs):
        tensor_ = tensor[i_song]
        mapped = (tonal_matrix @ tensor_).reshape((6, -1, n_tracks - 1))
        expanded1 = np.expand_dims(mapped, axis=-1)
        expanded2 = np.expand_dims(mapped, axis=-2)
        tonal_dist = np.linalg.norm(expanded1 - expanded2, axis=0)
        td[i_song] = np.sum(tonal_dist, axis=0)
    return td


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def for_train():
    n_tracks = 5
    n_measures = 4
    measure_resolution = 16
    n_pitches = 72

    with open("data/json/train_dict.json") as f:
        train_dict = json.load(f)
    eb = []
    upc = []
    dp = []
    td = []
    for song in train_dict.keys():
        for segment in train_dict[song]:
            pianoroll = np.load(segment)
            eb.append(empty_bars(pianoroll))
            upc.append(used_pitch_classes(pianoroll))
            dp.append(drum_pattern(pianoroll))
            td.append(tonal_distance(pianoroll))
    result = {}
    result["EB"] = {}
    result["UPC"] = {}
    result["DP"] = {}
    result["TD"] = {}

    result["EB"]["avg"] = np.average(eb, axis=(0, 1)).tolist()
    result["EB"]["var"] = np.var(eb, axis=(0, 1)).tolist()
    result["UPC"]["avg"] = np.average(upc, axis=(0, 1)).tolist()
    result["UPC"]["var"] = np.var(upc, axis=(0, 1)).tolist()
    result["DP"]["avg"] = np.average(dp, axis=(0, 1)).tolist()
    result["DP"]["var"] = np.var(dp, axis=(0, 1)).tolist()
    result["TD"]["avg"] = get_triu(np.average(td, axis=(0, 1))).tolist()
    result["TD"]["var"] = get_triu(np.var(td, axis=(0, 1))).tolist()

    with open("train_avg_var.json", "w") as f:
        json.dump(result, f)


def for_generated():
    n_tracks = 5
    n_measures = 4
    measure_resolution = 16
    n_pitches = 72
    output_dim = 64

    encoder = Encoder(n_tracks, n_measures, measure_resolution, n_pitches, output_dim)

    target_dirs = [
        "sotsuron_models/sotsuron1_d_conditioning_model",
        "sotsuron_models/sotsuron2_pianoroll_distance_model",
        "sotsuron_models/sotsuron3_embedding_distance_model",
        ]

    for dir in target_dirs:
        out_file_path = os.path.join(dir, "metrics.json")
        out_json = {}
        dirs = glob.glob(os.path.join(dir, "generated/*"))

        for song_dir in dirs:
            npys = glob.glob(os.path.join(song_dir, "npy", "*.npy"))
            out_json[song_dir] = {}

            for npy in npys:
                # sotsuron_models/sotsuron3_embedding_distance_model/generated/00005542_TRHIRGP128F4280974_repeat/npy/105.npy
                # TODO: ここのコードの汚さ…
                song_id = npy.split("/")[3]
                if (song_id[-7:] == "_repeat"):
                    song_id = song_id[:-7]
                elif (song_id[-6:] == "_noise"):
                    song_id = song_id[:-6]
                song_number = npy.split("/")[5]

                reference = np.load(os.path.join("data", "all", song_id, song_number))
                ref_embedding = encoder(torch.from_numpy(reference.astype(np.float32))).cpu().detach().numpy()

                out_json[song_dir][npy] = {}
                pianoroll = np.load(npy)
                out_json[song_dir][npy]["EB"] = empty_bars(pianoroll).tolist()
                out_json[song_dir][npy]["UPC"] = used_pitch_classes(pianoroll).tolist()
                out_json[song_dir][npy]["DP"] = drum_pattern(pianoroll).tolist()
                out_json[song_dir][npy]["TD"] = tonal_distance(pianoroll).tolist()

                distances = []
                with torch.inference_mode():
                    for i in range(pianoroll.shape[1] // (n_measures * measure_resolution)):
                        encoded = encoder(
                            torch.from_numpy(pianoroll[:, i * (n_measures * measure_resolution) : (i + 1) * (n_measures * measure_resolution)].astype(np.float32))
                            ).cpu().detach().numpy()[0]
                        distances.append(float(cos_sim(ref_embedding, encoded)[0]))
                out_json[song_dir][npy]["COS_SIM"] = distances

        with open(out_file_path, "w") as f:
            json.dump(out_json, f)


def process_avg_var(json_name):
    target_dirs = [
        "sotsuron_models/sotsuron1_d_conditioning_model",
        "sotsuron_models/sotsuron2_pianoroll_distance_model",
        "sotsuron_models/sotsuron3_embedding_distance_model",
        ]
    for dir in target_dirs:
        with open(os.path.join(dir, f"{json_name}.json")) as f:
            json_ = json.load(f)
        eb = []
        upc = []
        dp = []
        td = []
        cos_sim_ = []
        songs = list(json_.keys())
        for song in songs:
            for segment in json_[song].keys():
                eb.append(json_[song][segment]["EB"])
                upc.append(json_[song][segment]["UPC"])
                dp.append(json_[song][segment]["DP"])
                td.append(json_[song][segment]["TD"])
                cos_sim_.append(json_[song][segment]["COS_SIM"])
        eb = np.array(eb)
        upc = np.array(upc)
        dp = np.array(dp)
        td = np.array(td)
        cos_sim_ = np.array(cos_sim_)

        result = {}
        result["EB"] = {}
        result["UPC"] = {}
        result["DP"] = {}
        result["TD"] = {}
        result["COS_SIM"] = {}

        result["EB"]["avg"] = np.average(eb, axis=(0, 1)).tolist()
        result["EB"]["var"] = np.var(eb, axis=(0, 1)).tolist()
        result["UPC"]["avg"] = np.average(upc, axis=(0, 1)).tolist()
        result["UPC"]["var"] = np.var(upc, axis=(0, 1)).tolist()
        result["DP"]["avg"] = np.average(dp, axis=(0, 1)).tolist()
        result["DP"]["var"] = np.var(dp, axis=(0, 1)).tolist()
        result["TD"]["avg"] = get_triu(np.average(td, axis=(0, 1))).tolist()
        result["TD"]["var"] = get_triu(np.var(td, axis=(0, 1))).tolist()
        result["COS_SIM"]["avg"] = np.average(cos_sim_, axis=(0, 1)).tolist()
        result["COS_SIM"]["var"] = np.var(cos_sim_, axis=(0, 1)).tolist()

        with open(os.path.join(dir, f"{json_name}_avg_var.json"), "w") as f:
            json.dump(result, f)


def get_triu(ary):
    """2次元正方行列の，対角成分より上の要素を1次元に変換

    Args:
        ary (_type_): _description_

    Returns:
        _type_: _description_
    """
    new_ary = []
    dim = len(ary)
    for i in range(dim - 1):
        for j in range(dim - 1 - i):
            new_ary.append(ary[i][i + j + 1])

    return np.array(new_ary)


def avg_var():
    json_names = [
        "metrics_noise",
        "metrics_ref",
        "metrics_repeat",
        ]
    for i in json_names:
        process_avg_var(i)

if __name__ == "__main__":
    # tensor = np.load("outputs/sotsuron2/generated/s2_d_conditioning_f-conditioning_64-latent_64-adv_hinge-g_recon_L2_0-g_emb_COS_1_model/200000step/20230124-123444/generated.npy")
    # tensor = np.load("outputs/sotsuron/generated/d_conditioning_f-conditioning_64-latent_64-adv_hinge-g_recon_BCE_1_model/1000000step/20230110-115255/generated.npy")
    # print(empty_bars(tensor))
    # print(used_pitch_classes(tensor))
    # print(drum_pattern(tensor))
    # print(tonal_distance(tensor))

    # for_generated()
    # avg_var()

    for_train()
