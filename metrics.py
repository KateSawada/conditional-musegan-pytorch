import torch
import numpy as np


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

        dp[i_song] += np.sum(np.sum(tensor_, axis=1) * mask)
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

    print(tensor.shape)

    for i_song in range(n_songs):
        tensor_ = tensor[i_song]
        mapped = (tonal_matrix @ tensor_).reshape((6, -1, n_tracks - 1))
        expanded1 = np.expand_dims(mapped, axis=-1)
        expanded2 = np.expand_dims(mapped, axis=-2)
        tonal_dist = np.linalg.norm(expanded1 - expanded2, axis=0)
        td[i_song] = np.sum(tonal_dist, axis=0)
    return td



if __name__ == "__main__":
    tensor = np.load("outputs/sotsuron2/generated/s2_d_conditioning_f-conditioning_64-latent_64-adv_hinge-g_recon_L2_0-g_emb_COS_1_model/200000step/20230124-123444/generated.npy")
    tensor = np.load("outputs/sotsuron/generated/d_conditioning_f-conditioning_64-latent_64-adv_hinge-g_recon_BCE_1_model/1000000step/20230110-115255/generated.npy")
    print(empty_bars(tensor))
    print(used_pitch_classes(tensor))
    print(drum_pattern(tensor))
    print(tonal_distance(tensor))
