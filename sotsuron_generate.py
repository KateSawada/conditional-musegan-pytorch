import yaml
import itertools
import os

import generate
from custom import config


def main():
    d_conditioning = [True, False]
    conditioning_dim = [64, 128]
    latent_dim = [64, 128]
    adv = ["mse", "hinge"]
    g_recon = ["BCE", "L2"]
    g_recon_weight = [1, 10, 50]
    for i in itertools.product(d_conditioning, conditioning_dim, latent_dim,
                               adv, g_recon, g_recon_weight):
        dc = "t" if i[0] else "f"
        exp_name = f"d_conditioning_{dc}-conditioning_{i[1]}-latent_{i[2]}-adv_{i[3]}-g_recon_{i[4]}_{i[5]}"

        # ディレクトリがあれば生成実行
        if (os.path.exists(f"model/sotsuron/{exp_name}_model")):
            print(f"model/sotsuron/{exp_name}_model")
            config.load("outputs/tmp", [f"model/sotsuron/{exp_name}_model/save.yml"], initialize=True)
            # generate用のconfigを作成
            config["seed"] = 0
            config["generate_json"] = "data/json/sotsuron_test.json"
            config["generate_random"] = False
            config["conditioning_model"] = "triplet"
            config["conditioning_model_pth"] = f"ignore/conditioning/triplet/tested/dim{i[1]}/model1000.pth"
            config["n_samples"] = 8
            config["pth"] = f"model/sotsuron/{exp_name}_model/generator-final.pth"
            config["out_dir"] = f"outputs/{exp_name}_model"
            generate.generate("", config)


if __name__ == '__main__':
    main()
