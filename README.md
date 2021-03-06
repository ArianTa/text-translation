# Text Translation

> Project for the INFO2049-1 course of the ULiege University under the supervision of Professor Ittoo.

This project consists in a simple machine translation task.

The dataset is the [French-English Multi30k dataset](https://github.com/multi30k/dataset)
and the Word2Vec embeddings may be found [here](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/).

Other pretrained embeddings are downloadable through the torchtext library directly.

## Setup

### Download embedding vectors and dataset
The dataset and all the word embeddings used in this project may be manually downloaded [here](https://www.dropbox.com/s/lfzx0190ibz4dwx/text-translation.tar.gz?dl=1). Extract this archive in the local repository.

Or do it manually through the command line:
cd to your local repository
```sh
cd <local_repository>
```
Download the archive with curl
```sh
curl -L -o <archive_name>.tar.gz https://www.dropbox.com/s/lfzx0190ibz4dwx/text-translation.tar.gz?dl=1
```
Extract the archive
```sh
tar -xf <archive_name>.tar.gz
```
You can then remove the archive
```sh
rm <archive_name>.tar.gz
```

### Setup environment
Make sure you have an anaconda installation working on your machine.

The different packages needed are listed in the file `environment.yml` which can be used to create a new environment with the following instruction 
```sh
conda env create -f environment.yml -n <env_name>
```

Activate the environment
```sh
conda activate <env_name>
```

Install spacy models
```sh
python -m spacy download en && python -m spacy download fr
```

## Usage 

The entrypoint for training and testing is `main.py`.

### Configuration

The `config.py` file contains the architecture configuration as well as other hyperparameters.
- `TEST`: set to `True` for testing a model and to `False` otherwise
- `SAVE`: specify under which name the weights should be saved (irrelevant if `TEST` is set to `True`)
- `LOAD`: specify a path for a checkpoint to continue training or to test (relevant only if `TEST` is set to `True`)
- `EMB`: specify the word embeddings to be used (`FastText`, `GloVe` or `Word2Vec`, case insensitive)
- `BATCH_SIZE`: specify the batch size
- `N_EPOCHS`: specify the number of epochs
- `N_LAYERS`: specify the number of layers of both LSTMs of the encoder and decoder
- `HID_DIM`: specify the hidden state dimension of both LSTMs of the encoder and decoder. Note that it correspondes to the size of `h` and `c`, unconcatenated
- `ENC_DROPOUT`: dropout probability of the encoder
- `DEC_DROPOUT`: dropout probability of the decoder
- `FREEZE`: set to `True` to freeze the embedding layers during training and to `False` otherwise

### Train and test
Once the file `config.py` has been modified, just run
```sh
python3 main.py
```

## Meta
Authors: 
- Nora Folon - nfolon@student.ulg.ac.be
- Arian Tahiraj - atahiraj@student.ulg.ac.be
