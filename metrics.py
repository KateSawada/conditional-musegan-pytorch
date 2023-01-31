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

if __name__ == "__main__":
    tensor = np.load("outputs/sotsuron2/generated/s2_d_conditioning_f-conditioning_64-latent_64-adv_hinge-g_recon_L2_0-g_emb_COS_1_model/200000step/20230124-123444/generated.npy")
    tensor = np.load("outputs/sotsuron/generated/d_conditioning_f-conditioning_64-latent_64-adv_hinge-g_recon_BCE_1_model/1000000step/20230110-115255/generated.npy")
    print(empty_bars(tensor))
    print(used_pitch_classes(tensor))
