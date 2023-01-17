import os
import json
import itertools
import argparse
import glob

import numpy as np
import torch
from torch.nn.functional import normalize
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


from model import Encoder
from custom import config


def get_encoder(config):
    measure_resolution = 4 * config.beat_resolution
    encoder = Encoder(
        config.n_tracks, config.n_measures, measure_resolution, config.n_pitches,
        config.conditioning_dim)
    return encoder


def get_references(json_path):
    """jsonファイルに記載されたnpyファイルを読み込んでndarrayのリストを返す

    Args:
        json_path (str): jsonファイルへのパス

    Returns:
        list(tensor): ピアノロールのndarrayのlist
    """
    op = open(json_path, "r")
    songs_list = json.load(op)
    songs = []
    for song in songs_list:
        songs.append(torch.from_numpy(np.load(song).astype(np.float32)))
    return songs


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--models_dir", type=str, required=True,
    #                     help="models root directory")
    parser.add_argument("-s", "--samples_dir", type=str, required=True,
                        help="samples root directory")
    parser.add_argument("-g", "--gen_name", type=str, required=True,
                        help="gen name")
    # SANPLES_DIR/MODEL_NAME/GEN_NAME/TIMESTAMP/generated.npy
    parser.add_argument("-j", "--json_path", type=str, required=True,
                        help="reference songs json file")
    args = parser.parse_args()

    samples_dir = args.samples_dir
    gen_name = args.gen_name

    d_conditioning = [True, False]
    conditioning_dim = [64, 128]
    latent_dim = [64, 128]
    adv = ["mse", "hinge"]
    g_recon = ["BCE", "L2"]
    g_recon_weight = [1, 10, 50]

    # initialize color palette for plt
    models_per_graph = len(d_conditioning) * len(adv) * len(g_recon) * len(g_recon_weight)
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, models_per_graph))

    references = get_references(args.json_path)
    n_samples = 4
    n_songs = 2
    songs_markers = ["o", "^"]
    n_tracks = 5
    n_measures = 4
    beat_resolution = 4
    n_pitches = 72

    sample_timesteps = n_measures * 4 * beat_resolution

    # encoder と references は辞書に格納しておく
    encoders = {}
    reference_embeddings = {}
    for i in conditioning_dim:
        encoder = Encoder(
            n_tracks, n_measures, 4 * beat_resolution, n_pitches,i
            )
        encoder.load_state_dict(torch.load(f"ignore/conditioning/triplet/tested/dim{i}/model1000.pth"))
        encoders[i] = encoder
        reference_embeddings[i] = np.array([encoder(ref).to("cpu").detach().numpy().copy()[0] for ref in references])

    # データの格納先
    # data[conditioning_dim][latent_dim]
    #   > ndarray(shape=[models_per_graph, conditioning_dim])
    data = {}
    for i in conditioning_dim:
        data[i] = {}
        for j in latent_dim:
            data[i][j] = np.zeros([0, i])

    for i in itertools.product(d_conditioning, conditioning_dim, latent_dim,
                               adv, g_recon, g_recon_weight):
        dc = "t" if i[0] else "f"
        exp_name = f"d_conditioning_{dc}-conditioning_{i[1]}-latent_{i[2]}-adv_{i[3]}-g_recon_{i[4]}_{i[5]}"

        config.load("outputs/tmp", [f"model/sotsuron/{exp_name}_model/save.yml"], initialize=True)
        config["seed"] = 0
        config["generate_json"] = "data/json/sotsuron_test.json"
        config["generate_random"] = False
        config["conditioning_model"] = "triplet"
        config["conditioning_model_pth"] = f"ignore/conditioning/triplet/tested/dim{i[1]}/model1000.pth"

        # load npy
        # とりあえず見つかった末尾のものを使うように
        sample_path = glob.glob(os.path.join(samples_dir, exp_name + "_model", gen_name, "*", "generated.npy"))[-1]
        sample_npy = np.load(sample_path).astype(np.float32)

        # encode
        for j in range(0, sample_npy.shape[1], sample_timesteps):
            embedded = normalize(encoders[i[1]](torch.from_numpy(sample_npy[:, j : j + sample_timesteps, :])))

            data[i[1]][i[2]] = np.vstack((data[i[1]][i[2]], embedded.to("cpu").detach().numpy().copy()))

    fig, axs = plt.subplots(nrows=len(conditioning_dim), ncols=len(latent_dim), figsize=(10, 10))
    # 余白
    # plt.subplots_adjust(wspace=0.4, hspace=0.6)
    for i in range(len(conditioning_dim)):
        for j in range(len(latent_dim)):
            data_ = data[conditioning_dim[i]][latent_dim[j]]
            print(f"-- c {conditioning_dim[i]} l {latent_dim[j]} --")
            print(f"{data_.max(axis=0)}")
            print(f"{data_.min(axis=0)}")
            print(f"{data_.mean(axis=0)}")
            print(f"{data_.std(axis=0)}")
            # 次元圧縮
            t_sne_metrics = TSNE(n_components=2, random_state=0).fit_transform(np.vstack((data_, reference_embeddings[conditioning_dim[i]])))

            # plot
            axs[i, j].set_title(f"con {conditioning_dim[i]}, latent {latent_dim[j]}")
            for s in range(n_songs):
                for m in range(models_per_graph):
                    axs[i, j].scatter(
                        t_sne_metrics[m * n_samples * n_songs + s * n_samples: m * n_samples * n_songs + (s + 1) * n_samples, 0],
                        t_sne_metrics[m * n_samples * n_songs + s * n_samples: m * n_samples * n_songs + (s + 1) * n_samples, 1],
                        marker=songs_markers[s], color=colors[m], facecolor='none')
                # reference (黒)
                axs[i, j].scatter(
                    t_sne_metrics[-1 * n_samples * n_songs + s * n_samples - 1: -1 * n_samples * n_songs + (s + 1) * n_samples - 1, 0],
                    t_sne_metrics[-1 * n_samples * n_songs + s * n_samples - 1: -1 * n_samples * n_songs + (s + 1) * n_samples - 1, 1],
                    color="black", marker=songs_markers[s], label="reference")
            axs[i, j].legend()
            # axs[i, j].set_facecolor("none")
    # plt.savefig("outputs/sotsuron/figs/gen_and_ref_embeddings_normalized.png")
    plt.show()

if __name__ == '__main__':
    # $ python sotsuron_similarity.py -s outputs/sotsuron/generated -g 1000000step -j data/json/sotsuron_test.json
    main()
