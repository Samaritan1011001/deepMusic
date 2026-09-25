# Music Analysis — Genre Classification using Deep Learning

Classify music into genres directly from audio using deep learning. This project uses the
[Free Music Archive (FMA)](https://github.com/mdeff/fma) dataset, converts raw audio into
mel-spectrogram features, and trains a Convolutional Recurrent Neural Network (CRNN) to
predict the genre of a track.

## Table of Contents

- [Overview](#overview)
- [How It Works (Pipeline)](#how-it-works-pipeline)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
  - [1. Prerequisites](#1-prerequisites)
  - [2. Set up the environment](#2-set-up-the-environment)
  - [3. Configure the `.env` file](#3-configure-the-env-file)
  - [4. Get the data](#4-get-the-data)
  - [5. Train / evaluate](#5-train--evaluate)
- [Workflow](#workflow)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Results](#results)
- [Experiments & Trials](#experiments--trials)
- [Troubleshooting & GPU Notes](#troubleshooting--gpu-notes)
- [Documentation & Sources](#documentation--sources)
- [Contributing](#contributing)
- [License & Acknowledgements](#license--acknowledgements)

## Overview

Audio clips are pre-processed into mel-spectrograms and saved as `.npy` arrays for fast,
repeatable training. A CRNN (convolutional layers for local spectral patterns + recurrent
layers for temporal structure) is then trained on these features. The repository also
includes notebooks for visualizing what the network learns (filters, activations, and a
confusion matrix).

- **Input:** raw audio (FMA `small` and `medium` sets)
- **Features:** mel-spectrograms (via `librosa` / `python_speech_features`)
- **Model:** CRNN, trained weights stored in `data/weights/crnn_20_latest.h5`
- **Task:** multi-class genre classification

## How It Works (Pipeline)

The end-to-end pipeline turns raw audio files into genre predictions in a few stages:

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

```
raw audio (~30s, 44100 Hz)
        │  librosa / python_speech_features
        ▼
mel-spectrogram (96 × 1366, 1 channel)
        │  convert_to_npy.py
        ▼
.npy feature arrays
        │  cnn_model_using_npy_medium_set.ipynb
        ▼
CRNN model  ──►  genre prediction + metrics
```

## Model Architecture

The model is a **CRNN (Convolutional Recurrent Neural Network)**:

- **Convolutional front-end** — several convolutional/pooling blocks learn local
  time-frequency features from the mel-spectrogram input of shape `(96, 1366, 1)`.
- **Recurrent back-end** — recurrent layers model the temporal evolution of the learned
  features across the time axis of the spectrogram.
- **Classification head** — a dense output layer produces a probability over the target
  genres (`n_classes = 4` in the reference configuration).

Trained weights for the reference configuration are provided in
[`data/weights/crnn_20_latest.h5`](data/weights/crnn_20_latest.h5), so you can load them
and evaluate without retraining. The architecture diagram and the exact per-run
configurations are recorded under
[`docs/network_config_log/`](docs/network_config_log/).

## Dataset

The audio and metadata come from the FMA dataset. See [`data/dataset.txt`](data/dataset.txt)
for the exact links.

- **FMA data:** https://github.com/mdeff/fma — provides `fma_small.zip`, `fma_medium.zip`,
  and `fma_metadata.zip`.
- **Pre-processed `.npy` files:** used directly for training/testing so you can skip the
  heavy audio-processing step (link in `data/dataset.txt`).

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

## Getting Started

### 1. Prerequisites

- Python 3.x
- The Python packages listed in [`requirements.txt`](requirements.txt) (`numpy`, `pandas`,
  `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`, `keras`, `librosa`, `requests`,
  `pydot`, `tqdm`, `jupyter`, `python-dotenv`, `python_speech_features`).
- Optional but recommended: an NVIDIA GPU with a matching CUDA/cuDNN setup for TensorFlow —
  see [Troubleshooting & GPU Notes](#troubleshooting--gpu-notes).

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
[`data/dataset.txt`](data/dataset.txt) and place them under `data/` as described in the
project structure below. Note that the raw data directories (`data/fma_small`,
`data/npy_files`), `*.mp3`, `*.csv`, and `.env` are git-ignored, so you must download the
data yourself — it is not part of the repository.

### 5. Train / evaluate

Open the core notebook and run the cells:

```bash
jupyter notebook src/cnn_model_using_npy_medium_set.ipynb
```

> **Note:** The config variables at the top of the notebooks/scripts must be set to select
> either the **small** or **medium** dataset for training, plotting, etc.

## Workflow

1. **Clean metadata & audio** — `src/processing/clean_audio_files.py`
2. **Pre-process to features** — `src/processing/convert_to_npy.py`
   (or `pre_process_colab.py` when running on Google Colab)
3. **Train the CRNN** — `src/cnn_model_using_npy_medium_set.ipynb`
4. **Visualize results** — `src/visualize.ipynb`

## Configuration

Key feature/model parameters live in the pre-processing generator params (`gen_params` in
[`src/processing/convert_to_npy.py`](src/processing/convert_to_npy.py)) and in the audio
constants in [`sources/utils.py`](sources/utils.py):

| Parameter        | Value          | Meaning                                                    |
|------------------|----------------|------------------------------------------------------------|
| `dim`            | `(96, 1366)`   | Mel-spectrogram shape (mel bands × time frames)            |
| `n_channels`     | `1`            | Single-channel (grayscale) spectrogram input               |
| `n_classes`      | `4`            | Number of target genres in the reference configuration     |
| `batch_size`     | `10`           | Samples per training batch                                 |
| `SAMPLING_RATE`  | `44100`        | Audio sampling rate in Hz                                  |
| `NB_AUDIO_SAMPLES` | `1321967`    | Samples per ~30-second clip                                |

When switching between the `small` and `medium` sets, update the dataset config variable at
the top of the relevant script/notebook (and/or the `.env` file) so that data loading,
plotting, and training all point at the same subset.

## Project Structure

```
deepMusic/
├── data/
│   ├── fma_metadata/     # metadata needed to load audio files (track ids, genres)
│   ├── fma_small/        # small FMA dataset (download — see data/dataset.txt)
│   ├── npy_files/         # processed features for small & medium sets (download)
│   └── weights/           # trained weights (crnn_20_latest.h5)
├── docs/                  # graphs, report, logs, and network config results
│   ├── graphs/            # spectrograms, class distributions, visualizations
│   └── network_config_log/# per-config results and architecture diagrams
├── fma_env/               # virtual environment for the project
├── sources/               # utilities (utils.py, setup.py, tf_check.py) and references
├── src/
│   ├── helper/            # helper scripts (e.g. plot class distribution)
│   ├── processing/        # pre-processing, metadata cleaning, npy conversion
│   ├── trials/            # experimental network configurations
│   ├── cnn_model_using_npy_medium_set.ipynb  # core: loads npy files & trains the network
│   └── visualize.ipynb    # produces network visualizations
├── requirements.txt
└── README.md
```

## Results

Example evaluation on a 4-genre medium subset (Electronic, Folk, Rock, Hip-Hop),
from [`docs/network_config_log/Results.txt`](docs/network_config_log/Results.txt):

| Genre       | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| Electronic  | 0.70      | 0.67   | 0.69     | 530     |
| Folk        | 0.00      | 0.00   | 0.00     | 52      |
| Rock        | 0.00      | 0.00   | 0.00     | 119     |
| Hip-Hop     | 0.69      | 0.91   | 0.79     | 609     |
| **Accuracy**|           |        | **0.70** | 1310    |

The imbalance in the dataset (many more Electronic/Hip-Hop samples than Folk/Rock) is
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

## Troubleshooting & GPU Notes

- **Check TensorFlow / GPU visibility.** Run [`sources/tf_check.py`](sources/tf_check.py) to
  confirm TensorFlow is installed and can see your GPU before launching a long training run.
- **GPU memory growth.** The training code enables TensorFlow GPU handling; if you hit
  out-of-memory errors, reduce `batch_size` in the generator params or run on a smaller
  subset (the `small` set) first.
- **No GPU?** The pipeline still runs on CPU, just more slowly. Prefer the pre-processed
  `.npy` files and the `small` set for CPU-only experimentation.
- **`.env` not loaded / paths not found.** Make sure a `.env` file exists in the project
  root and that `python-dotenv` is installed (it is in `requirements.txt`). Missing or
  wrong paths here are the most common cause of data-loading errors.
- **`librosa` / audio decoding errors.** `librosa` may require a backend such as `ffmpeg`
  to decode `.mp3` files; install it via your OS package manager if audio loading fails.

## Documentation & Sources

- `docs/` — training graphs, mel-spectrograms per genre, logs, and the project report.
  - `docs/graphs/` — signals, FFT, filter banks, MFCCs, per-genre mel-spectrograms, and
    class-distribution plots for the small and medium sets.
  - `docs/graphs/visualization/` — confusion matrix and visualizations of learned
    convolutional filters and activations.
  - `docs/network_config_log/` — per-configuration results (`Results.txt`), config logs,
    and architecture diagrams.
- `sources/` — utilities (`utils.py`, `setup.py`, `tf_check.py`) and references that
  informed the approach.

## Contributing

Contributions are welcome. A typical flow:

1. Fork the repository and create a feature branch.
2. Make your changes, keeping the pipeline stages (pre-processing → training →
   visualization) intact.
3. If you change feature dimensions or the number of classes, update the
   [Configuration](#configuration) table so the docs stay accurate.
4. Open a pull request describing the change and, where relevant, include updated
   result logs under `docs/network_config_log/`.

## License & Acknowledgements

This repository does not currently include an explicit license file; if you intend to reuse
the code, please contact the repository owner regarding usage terms.

Acknowledgements:

- **Free Music Archive (FMA)** — the dataset and loading tooling come from
  [mdeff/fma](https://github.com/mdeff/fma). Please follow the FMA project's own terms and
  cite it if you use the data.
- Built with open-source tools including TensorFlow/Keras, librosa, scikit-learn, pandas,
  NumPy, matplotlib, and seaborn.
