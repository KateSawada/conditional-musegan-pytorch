# setup
```sh
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ sh data_preparation.sh
```

# train
```sh
$ python train.py -c config/train.yml config/base.yml
```

# generate

```sh
$ python generate.py -c config/generate.yml config/base.yml -m model/trained
```
