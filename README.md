# Music Analysis — Genre Classification using Deep Learning

Classify music into genres directly from audio using deep learning. This project uses the
[Free Music Archive (FMA)](https://github.com/mdeff/fma) dataset, converts raw audio into
mel-spectrogram features, and trains a Convolutional Recurrent Neural Network (CRNN) to
predict the genre of a track.

The repository contains the full pipeline: metadata cleaning, audio-to-feature conversion,
model definition and training, evaluation, and a set of notebooks for visualizing what the
network learns (convolution filters, activations, and a confusion matrix).

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
  - [Small vs. medium sets](#small-vs-medium-sets)
  - [Class imbalance & class weighting](#class-imbalance--class-weighting)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
  - [1. Clone the repository](#1-clone-the-repository)
  - [2. Set up the environment](#2-set-up-the-environment)
  - [3. Configure the `.env` file](#3-configure-the-env-file)
  - [4. Get the data](#4-get-the-data)
  - [5. Configure the run](#5-configure-the-run)
  - [6. Train and evaluate](#6-train-and-evaluate)
- [Pipeline in Detail](#pipeline-in-detail)
  - [Step 1 — Clean the metadata and audio](#step-1--clean-the-metadata-and-audio)
  - [Step 2 — Convert audio to mel-spectrogram features](#step-2--convert-audio-to-mel-spectrogram-features)
  - [Step 3 — Train the CRNN](#step-3--train-the-crnn)
  - [Step 4 — Visualize the model](#step-4--visualize-the-model)
- [Feature Extraction Details](#feature-extraction-details)
- [Model Architecture](#model-architecture)
- [Configuration Reference](#configuration-reference)
- [Results](#results)
- [Experiments & Trials](#experiments--trials)
- [Project Structure](#project-structure)
- [Notebooks and Scripts](#notebooks-and-scripts)
- [Running on Google Colab](#running-on-google-colab)
- [Troubleshooting & GPU Notes](#troubleshooting--gpu-notes)
- [Documentation and Sources](#documentation-and-sources)
- [Contributing](#contributing)
- [License & Acknowledgements](#license--acknowledgements)

---

## Overview

Audio clips are pre-processed into mel-spectrograms and saved as `.npy` arrays for fast,
repeatable training. A CRNN (convolutional layers capture local spectral patterns and
recurrent layers model temporal structure) is then trained on these features to predict a
track's top-level genre. The repository also includes notebooks for visualizing what the
network learns (filters, activations, and a confusion matrix).

- **Input:** raw audio from the FMA `small` and `medium` sets (MP3).
- **Features:** log-scaled mel-spectrograms of shape `(1, 96, 1366)` (1 channel, 96 mel
  bins, 1366 time frames).
- **Model:** a `MusicTaggerCRNN`-style network (3 convolutional blocks followed by 2 GRU
  layers and dense classification layers).
- **Task:** multi-class genre classification over four genres —
  `Electronic`, `Folk`, `Rock`, `Hip-Hop`.
- **Trained weights:** stored in `data/weights/crnn_20_latest.h5`.

## How It Works

```
raw audio (mp3)
      │
      ▼
clean_audio_files.py     # drop tracks librosa cannot decode, write cleaned_<subset>.csv
      │
      ▼
convert_to_npy.py        # load audio, compute mel-spectrograms, save x/y .npy arrays
      │
      ▼
cnn_model_using_npy_medium_set.ipynb   # load .npy arrays, build + train the CRNN
      │
      ▼
crnn_20_latest.h5        # trained weights
      │
      ▼
visualize.ipynb          # confusion matrix, learned filters and activations
```

The heavy audio-processing work is done once by `convert_to_npy.py`, which serializes the
feature tensors to `.npy` files. Training then loads those arrays directly, so you can
iterate on the network without re-decoding audio every time. Pre-processed `.npy` files are
also available for download (see [Dataset](#dataset)) so you can skip the audio step entirely.

In short, the pipeline turns raw audio into genre predictions in a few stages:

1. **Raw audio.** Each FMA track is a ~30-second clip. At the sampling rate of
   **44,100 Hz** used by the project (`SAMPLING_RATE` in
   [`sources/utils.py`](sources/utils.py)), a clip contains roughly **1,321,967 samples**
   (`NB_AUDIO_SAMPLES`).
2. **Feature extraction (mel-spectrograms).** Each clip is converted into a
   mel-spectrogram with a shape of **(96, 1366)** — 96 mel-frequency bands over 1366 time
   frames. This 2-D time-frequency representation is what the network actually "sees".
3. **Serialization to `.npy`.** The spectrograms are stored as NumPy `.npy` arrays so the
   expensive audio-processing step only has to run once. Subsequent training and evaluation
   load these arrays directly.
4. **Model training.** A CRNN consumes the mel-spectrograms as single-channel
   (`n_channels = 1`) images, learns local spectral patterns with convolutional layers, and
   captures temporal structure with recurrent layers.
5. **Prediction & evaluation.** The trained model outputs a genre for each clip. Results are
   summarized with per-class precision/recall/F1 and a confusion matrix.

## Dataset

The audio and metadata come from the FMA dataset. See [`data/dataset.txt`](data/dataset.txt)
for the exact links.

- **FMA data:** [github.com/mdeff/fma](https://github.com/mdeff/fma) — provides
  `fma_small.zip`, `fma_medium.zip`, and `fma_metadata.zip`. The metadata archive contains
  `tracks.csv`, `features.csv`, and `echonest.csv`, which map track IDs to genres and splits.
- **Pre-processed `.npy` files:** the processed feature arrays used directly for
  training and testing, so you can skip the heavy audio-processing step. The download link
  is in [`data/dataset.txt`](data/dataset.txt).

The FMA metadata already defines `training`, `validation`, and `test` splits, and this
project reuses those splits rather than re-partitioning the data.

### Small vs. medium sets

The project supports two FMA subsets, and most scripts/notebooks have a config variable at
the top to switch between them:

- **`small`** — a smaller, more balanced subset that is quick to download and iterate on.
  Good for smoke-testing the pipeline end to end.
- **`medium`** — a larger subset used for the reference results below. It is more realistic
  but also more **imbalanced** across genres.

### Class imbalance & class weighting

The medium subset contains many more Electronic/Hip-Hop samples than Folk/Rock samples.
This imbalance is visible in the class-distribution plots under
[`docs/graphs/`](docs/graphs/) and directly affects per-class performance. To partially
compensate, the pre-processing/training code computes class weights with scikit-learn's
`compute_class_weight` (via a `LabelEncoder`) so that under-represented genres contribute
more to the loss. Even so, heavily under-represented classes can still score poorly — see
[Results](#results).

## Requirements

- Python 3.x
- The Python packages listed in [`requirements.txt`](requirements.txt):
  `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`, `keras`,
  `librosa`, `requests`, `pydot`, `tqdm`, `jupyter`, `python-dotenv`, and
  `python_speech_features`.
- **FFmpeg** must be installed and on your `PATH`. The audio loaders in
  [`sources/utils.py`](sources/utils.py) (notably `FfmpegLoader`) shell out to `ffmpeg`, and
  `librosa` also relies on it to decode MP3 files.
- A GPU is recommended for training but not required. The notebooks include a small
  compatibility shim (`_get_available_gpus`) so Keras can detect logical devices on newer
  TensorFlow versions. See [Troubleshooting & GPU Notes](#troubleshooting--gpu-notes).

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Samaritan1011001/deepMusic.git
cd deepMusic
```

### 2. Set up the environment

```bash
# create and activate a virtual environment
python -m venv fma_env
source fma_env/bin/activate      # on Windows: fma_env\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

### 3. Configure the `.env` file

The utilities in [`sources/utils.py`](sources/utils.py) read configuration from a `.env`
file using [`python-dotenv`](https://pypi.org/project/python-dotenv/). Create a `.env` file
in the project root with the paths/values your setup needs (for example the base data
directory and the selected dataset). The `.env` file is intentionally git-ignored (see
[`.gitignore`](.gitignore)) so local paths and secrets are never committed.

```bash
# .env (example — adjust to your machine)
DATA_DIR=./data
DATASET=medium
```

### 4. Get the data

Download the FMA data and/or the pre-processed `.npy` files using the links in
[`data/dataset.txt`](data/dataset.txt) and place them under `data/` following the layout in
[Project Structure](#project-structure):

- Unzip `fma_metadata.zip` into `data/fma_metadata/`.
- Unzip `fma_small.zip` / `fma_medium.zip` into `data/fma_small/` / `data/fma_medium/` if
  you plan to regenerate features from audio.
- Place the downloaded `.npy` files under `data/npy_files/<subset>/` if you want to skip the
  audio-processing step.

Note that the raw data directories (`data/fma_small`, `data/npy_files`), `*.mp3`, `*.csv`,
and `.env` are git-ignored, so you must download the data yourself — it is not part of the
repository.

### 5. Configure the run

Each script and notebook has a small config block near the top. Two settings matter most:

- `root_folder` — the absolute path to your local copy of the repository. The scripts are
  checked in with a Windows-style path (`D:\...`) and a commented Linux alternative, so
  **update this to your own path** before running.
- `subset_config` — selects the dataset:

  ```python
  subset_config = {
      "audio_dir": "fma_medium",   # or "fma_small"
      "subset":    "medium"        # or "small"
  }
  ```

See [Configuration Reference](#configuration-reference) for the full list of knobs.

### 6. Train and evaluate

Open the core notebook and run the cells top to bottom:

```bash
jupyter notebook src/cnn_model_using_npy_medium_set.ipynb
```

The notebook loads the `.npy` feature arrays, builds the CRNN, trains it, plots the
training/validation loss and accuracy curves, and prints the test accuracy along with a
classification report and confusion matrix.

> **Note:** the config variables at the top of the notebooks/scripts must be set to select
> either the **small** or **medium** dataset for training, plotting, and evaluation.

## Pipeline in Detail

### Step 1 — Clean the metadata and audio

Script: [`src/processing/clean_audio_files.py`](src/processing/clean_audio_files.py)

- Loads `data/fma_metadata/tracks.csv` and selects the configured subset.
- Attempts to decode every track with `librosa.load(...)`. Any track that raises an error
  (corrupted or unreadable audio) is collected and dropped.
- Writes a cleaned metadata CSV (`data/fma_metadata/cleaned_<subset>.csv`) that downstream
  steps consume. The `to_csv` write is commented out by default — **uncomment it** to
  persist the cleaned file.

### Step 2 — Convert audio to mel-spectrogram features

Script: [`src/processing/convert_to_npy.py`](src/processing/convert_to_npy.py)
(or [`pre_process_colab.py`](src/processing/pre_process_colab.py) when running on Colab)

- Reads the cleaned metadata and filters to the four target genres
  (`Electronic`, `Folk`, `Rock`, `Hip-Hop`).
- Uses the FMA `training` / `validation` / `test` splits from the metadata.
- Computes balanced class weights with `scikit-learn` to counter class imbalance.
- Streams audio through a Keras `Sequence` `DataGenerator`, computing a mel-spectrogram per
  clip via `compute_melgram(...)`.
- Saves `x_train.npy`, `y_train.npy`, `x_val.npy`, `y_val.npy`, `x_test.npy`, and
  `y_test.npy` under `data/npy_files/<subset>/`. The `np.save` calls are commented out by
  default under an `UNCOMMENT TO SAVE` marker — **uncomment them** to persist the arrays.

`pre_process_colab.py` additionally produces exploratory plots (time-series signals, FFTs,
filter-bank coefficients, MFCCs, per-genre mel-spectrograms, and a class-distribution pie
chart) and saves them under `docs/graphs/<subset>/`.

### Step 3 — Train the CRNN

Notebook: [`src/cnn_model_using_npy_medium_set.ipynb`](src/cnn_model_using_npy_medium_set.ipynb)

- Loads the pre-computed `.npy` arrays and reshapes them into
  `(samples, 1, 96, 1366)` tensors (Keras `channels_first` ordering).
- Builds the CRNN via `MusicTaggerCRNN(weights=None, input_tensor=(1, 96, 1366))`.
- Compiles with the Adam optimizer (`learning_rate=0.0001`) and
  `categorical_crossentropy` loss, then trains for 20 epochs.
- Plots training/validation loss and accuracy, evaluates on the test set, and (optionally)
  saves the model with `model.save(...)`.

The notebook contains a second, commented-out variant of the architecture tuned for the FMA
**small** dataset (different pooling strides) under the `USE IT FOR FMA SMALL DATASET` marker.

### Step 4 — Visualize the model

Notebook: [`src/visualize.ipynb`](src/visualize.ipynb)

- Loads a trained model and the test `.npy` arrays.
- Produces a confusion matrix and classification report.
- Uses [`keras-vis`](https://github.com/raghakot/keras-vis) to visualize learned
  convolution filters (e.g. `conv1`, `conv3`) and activation maximizations, rendered as
  mel-spectrograms. Outputs are saved under `docs/graphs/visualization/`.

## Feature Extraction Details

Mel-spectrograms are computed by `compute_melgram(...)` with these parameters:

| Parameter          | Value   | Meaning                                             |
|--------------------|---------|-----------------------------------------------------|
| `SR`               | 12000   | target sample rate (Hz)                             |
| `N_FFT`            | 512     | FFT window size                                     |
| `N_MELS`           | 96      | number of mel bins (feature height)                 |
| `HOP_LEN`          | 256     | hop length between frames                           |
| `DURA`             | 29.12 s | clip duration, chosen to yield exactly 1366 frames  |

Each clip is trimmed or zero-padded to `DURA * SR` samples so that every mel-spectrogram has
the same shape. The spectrogram is converted to decibels with
`librosa.amplitude_to_db(... ** 2, ref=np.max)` and reshaped to `(1, 1, 96, 1366)`.

For exploratory analysis, `pre_process_colab.py` also computes filter-bank coefficients and
MFCCs via `python_speech_features` (`logfbank`, `mfcc`).

## Model Architecture

`MusicTaggerCRNN` (defined in the training notebook) uses Keras with `channels_first`
image ordering and an input shape of `(1, 96, 1366)`:

1. **Conv block 1** — `Conv2D(16, (2, 2), 'same', selu, lecun_normal)` →
   `BatchNorm` → `MaxPool(3, 3)` → `AlphaDropout(0.1)`
2. **Conv block 2** — `Conv2D(32, (2, 2), 'same', selu, lecun_normal)` →
   `BatchNorm` → `MaxPool(3, 3)` → `AlphaDropout(0.1)`
3. **Conv block 3** — `Conv2D(32, (3, 3), 'same', selu, lecun_normal)` →
   `BatchNorm` → `MaxPool(3, 3)` → `AlphaDropout(0.1)`
4. **Reshape** — permute and reshape the feature maps into a time-major sequence.
5. **Recurrent block** — `GRU(8, return_sequences=True)` →
   `GRU(8, return_sequences=False)` → `AlphaDropout(0.3)`
6. **Classifier** — `Dense(128, relu)` → `Dense(4, softmax)`

Design notes:

- The `selu` activation paired with `lecun_normal` initialization and `AlphaDropout` follows
  the self-normalizing network recipe.
- The function also supports a transfer-learning path (`weights='msd'`) that loads
  Million Song Dataset weights, pops the final layers, and attaches a new 4-class head.
- The small-dataset variant uses different pooling strides (see the commented cell in the
  training notebook).

Trained weights for the reference configuration are provided in
[`data/weights/crnn_20_latest.h5`](data/weights/crnn_20_latest.h5), so you can load them and
evaluate without retraining. The architecture diagram and the exact per-run configurations
are recorded under [`docs/network_config_log/`](docs/network_config_log/).

## Configuration Reference

These variables appear at the top of the scripts and notebooks:

| Variable                       | Where                          | Purpose                                              |
|--------------------------------|--------------------------------|------------------------------------------------------|
| `root_folder`                  | all scripts/notebooks          | absolute path to your local repository               |
| `subset_config["audio_dir"]`   | processing + training          | `fma_small` or `fma_medium`                          |
| `subset_config["subset"]`      | processing + training          | `small` or `medium`                                  |
| `gen_params["dim"]`            | processing + training          | feature dimensions, `(96, 1366)`                     |
| `gen_params["batch_size"]`     | processing                     | data-generator batch size (10)                       |
| `gen_params["n_classes"]`      | processing + training          | number of genres (4)                                 |
| `gen_params["n_channels"]`     | processing + training          | input channels (1)                                   |
| `network_config["optimizer"]`  | training notebook              | `Adam(learning_rate=0.0001)`                         |
| `network_config["loss"]`       | training notebook              | `categorical_crossentropy`                           |
| `network_config["epochs"]`     | training notebook              | training epochs (20)                                 |
| `network_config["batch_size"]` | training notebook              | training batch size (128)                            |

The core audio constants (`SAMPLING_RATE = 44100`, `NB_AUDIO_SAMPLES = 1321967`) live in
[`sources/utils.py`](sources/utils.py). When switching between the `small` and `medium` sets,
update the dataset config variable at the top of the relevant script/notebook (and/or the
`.env` file) so that data loading, plotting, and training all point at the same subset.

## Results

Example evaluation on a 4-genre medium subset (Electronic, Folk, Rock, Hip-Hop),
from [`docs/network_config_log/Results.txt`](docs/network_config_log/Results.txt):

| Genre        | Precision | Recall | F1-score | Support |
|--------------|-----------|--------|----------|---------|
| Electronic   | 0.70      | 0.67   | 0.69     | 530     |
| Folk         | 0.00      | 0.00   | 0.00     | 52      |
| Rock         | 0.00      | 0.00   | 0.00     | 119     |
| Hip-Hop      | 0.69      | 0.91   | 0.79     | 609     |
| **Accuracy** |           |        | **0.70** | 1310    |

Confusion matrix (rows = true, columns = predicted; order Electronic, Folk, Rock, Hip-Hop):

```
[[354   0   0 176]
 [  4   0   0  48]
 [ 93   0   0  26]
 [ 52   0   0 557]]
```

The strong class imbalance (many more Electronic/Hip-Hop samples than Folk/Rock) is
reflected in the per-class scores: Folk and Rock score `0.00` because the model rarely
predicts these heavily under-represented classes. Overall accuracy is **0.70** across 1310
samples. See `docs/graphs/` for class-distribution plots and
`docs/graphs/visualization/` for the confusion matrix and learned filters/activations.
Addressing the imbalance further (more data, stronger class weighting, resampling, or
data augmentation) is the main lever for improving the weak classes.

## Experiments & Trials

The [`src/trials/`](src/trials/) folder holds alternative and experimental configurations
that were explored while developing the reference model:

- **`cnn_model_colab.ipynb`** — a Google Colab version of the training notebook.
- **`cnn_model_using_generator_small_set.ipynb`** — trains on the small set using a Keras
  data generator (features produced on the fly rather than pre-computed `.npy` files).
- **`cnn_model_using_npy_medium_set_comet_version.py`** — a script variant of the core
  training run instrumented with [Comet](https://www.comet.com/) for experiment tracking.
- **`trial1_only_cnns_small.py`** — an early CNN-only baseline (no recurrent layers) on the
  small set.

Per-run configurations, architecture diagrams, and result logs are recorded under
[`docs/network_config_log/`](docs/network_config_log/), and training curves for successive
trials are under [`docs/graphs/`](docs/graphs/).

## Project Structure

```
deepMusic/
├── data/
│   ├── fma_metadata/      # metadata (tracks.csv, features.csv, echonest.csv, cleaned_<subset>.csv)
│   ├── fma_small/         # small FMA audio (download — see data/dataset.txt)
│   ├── fma_medium/        # medium FMA audio (download — see data/dataset.txt)
│   ├── npy_files/         # processed feature arrays for small & medium sets (download)
│   ├── weights/           # trained weights (crnn_20_latest.h5)
│   └── dataset.txt        # download links for the FMA data and processed .npy files
├── docs/
│   ├── graphs/            # spectrograms, class distributions, training logs, visualizations
│   └── network_config_log/# per-config results (Results.txt) and architecture diagrams
├── fma_env/               # virtual environment for the project
├── sources/               # FMA utilities (utils.py), setup.py, environment check (tf_check.py)
├── src/
│   ├── helper/            # helper scripts (e.g. plot_class_dist_medium.py)
│   ├── processing/        # metadata cleaning, npy conversion, Colab pre-processing
│   ├── trials/            # experimental network configurations
│   ├── cnn_model_using_npy_medium_set.ipynb  # core: loads npy files & trains the network
│   └── visualize.ipynb    # produces network visualizations
├── requirements.txt
└── README.md
```

## Notebooks and Scripts

| Path                                                    | Role                                                        |
|---------------------------------------------------------|-------------------------------------------------------------|
| `src/cnn_model_using_npy_medium_set.ipynb`              | Core notebook: load `.npy`, build/train/evaluate the CRNN.  |
| `src/visualize.ipynb`                                   | Confusion matrix and learned filter/activation visuals.     |
| `src/processing/clean_audio_files.py`                   | Drop undecodable tracks, write `cleaned_<subset>.csv`.      |
| `src/processing/convert_to_npy.py`                      | Compute mel-spectrograms and save feature arrays.           |
| `src/processing/pre_process_colab.py`                   | Colab pre-processing plus exploratory feature plots.        |
| `src/helper/plot_class_dist_medium.py`                  | Plot class distributions for the medium subset splits.      |
| `src/trials/cnn_model_colab.ipynb`                      | Experimental Colab training notebook.                       |
| `src/trials/cnn_model_using_generator_small_set.ipynb`  | Trial: train from a data generator on the small set.        |
| `src/trials/cnn_model_using_npy_medium_set_comet_version.py` | Trial: medium-set training with Comet.ml logging.      |
| `src/trials/trial1_only_cnns_small.py`                  | Trial: CNN-only baseline on the small set.                  |
| `sources/utils.py`                                      | FMA helpers: metadata `load`, `get_audio_path`, audio loaders. |

## Running on Google Colab

Several notebooks were developed on Colab and include cells such as:

```python
!pip install python_speech_features
!pip install python-dotenv
!pip install keras --upgrade
```

They also append a Google Drive path to `sys.path` and read data from Drive, for example:

```python
import sys
sys.path.append('/content/drive/My Drive/cs577- Deep learning/deepMusic/')
```

When running on Colab, mount your Drive, place the dataset under a Drive folder, and update
`root_folder` / the `sys.path` entry to match your Drive layout. Use
`src/processing/pre_process_colab.py` for the Colab pre-processing path.

## Troubleshooting & GPU Notes

- **`ffmpeg` not found / MP3 fails to load** — install FFmpeg and ensure it is on your
  `PATH`. The `FfmpegLoader` and `librosa` both depend on it.
- **Wrong paths** — every script ships with a Windows-style `root_folder`. Update it to your
  absolute repository path (there is a commented Linux example in the files).
- **`.env` not loaded / paths not found** — make sure a `.env` file exists in the project
  root and that `python-dotenv` is installed (it is in `requirements.txt`). Missing or wrong
  paths here are a common cause of data-loading errors.
- **No `.npy` files** — either download the processed arrays or run `convert_to_npy.py` with
  the `np.save` lines uncommented to generate them.
- **Check TensorFlow / GPU visibility** — run [`sources/tf_check.py`](sources/tf_check.py) to
  confirm TensorFlow is installed and can see your GPU before launching a long training run.
- **Keras cannot find GPUs on newer TensorFlow** — the notebooks patch
  `tfback._get_available_gpus` for compatibility; keep that cell if you upgrade TensorFlow.
- **GPU out-of-memory** — reduce `batch_size` in the generator params or run on the smaller
  `small` subset first.
- **No GPU?** — the pipeline still runs on CPU, just more slowly. Prefer the pre-processed
  `.npy` files and the `small` set for CPU-only experimentation.
- **`channels_first` errors** — the model expects `channels_first` ordering
  (`K.set_image_data_format('channels_first')`); do not switch to `channels_last`.

## Documentation and Sources

- `docs/` — training graphs, per-genre mel-spectrograms, class-distribution plots, logs, and
  the project report.
  - `docs/graphs/` — signals, FFT, filter banks, MFCCs, per-genre mel-spectrograms, and
    class-distribution plots for the small and medium sets.
  - `docs/graphs/visualization/` — confusion matrix and visualizations of learned
    convolutional filters and activations.
  - `docs/network_config_log/` — per-configuration results (`Results.txt`), config logs, and
    architecture diagrams.
- `sources/` — research papers and references that informed the approach, plus the FMA
  utility module (`utils.py`), `setup.py`, and the environment check (`tf_check.py`) used
  throughout the code.

## Contributing

Contributions are welcome. A typical flow:

1. Fork the repository and create a feature branch.
2. Make your changes, keeping the pipeline stages (pre-processing → training →
   visualization) intact.
3. If you change feature dimensions or the number of classes, update the
   [Configuration Reference](#configuration-reference) table so the docs stay accurate.
4. Open a pull request describing the change and, where relevant, include updated
   result logs under `docs/network_config_log/`.

## License & Acknowledgements

This repository does not currently include an explicit license file; if you intend to reuse
the code, please contact the repository owner regarding usage terms.

Acknowledgements:

- The [Free Music Archive (FMA)](https://github.com/mdeff/fma) dataset and its utility code
  by Michaël Defferrard et al. Please follow the FMA project's own terms and cite it if you
  use the data.
- The `MusicTaggerCRNN` architecture is adapted from Keras audio-tagging examples and the
  associated music auto-tagging research.
- Built with open-source tools including TensorFlow/Keras, librosa, scikit-learn, pandas,
  NumPy, matplotlib, and seaborn.
