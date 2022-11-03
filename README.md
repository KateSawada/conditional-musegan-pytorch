# setup
```sh
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ sh data_preparation.sh
```

# model preview on Tensorboard
```sh
python preview.py -c config/train.yml config/base.yml -m model
```

# train
```sh
$ python train.py -c config/train.yml config/base.yml -m model
```

# generate

```sh
$ python generate.py -c config/generate.yml config/base.yml -m model/trained
```
