# Music Analysis — Genre Classification using Deep Learning

Classify music into genres directly from audio using deep learning. This project uses the
[Free Music Archive (FMA)](https://github.com/mdeff/fma) dataset, converts raw audio into
mel-spectrogram features, and trains a Convolutional Recurrent Neural Network (CRNN) to
predict the genre of a track.

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

## Dataset

The audio and metadata come from the FMA dataset. See [`data/dataset.txt`](data/dataset.txt)
for the exact links.

- **FMA data:** https://github.com/mdeff/fma — provides `fma_small.zip`, `fma_medium.zip`,
  and `fma_metadata.zip`.
- **Pre-processed `.npy` files:** used directly for training/testing so you can skip the
  heavy audio-processing step (link in `data/dataset.txt`).

## Getting Started

### 1. Prerequisites

- Python 3.x
- The Python packages listed in [`requirements.txt`](requirements.txt) (`numpy`, `pandas`,
  `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`, `keras`, `librosa`, `requests`,
  `pydot`, `tqdm`, `jupyter`, `python-dotenv`, `python_speech_features`).

### 2. Set up the environment

```bash
# create and activate a virtual environment
python -m venv fma_env
source fma_env/bin/activate      # on Windows: fma_env\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

### 3. Get the data

Download the FMA data and/or the pre-processed `.npy` files using the links in
[`data/dataset.txt`](data/dataset.txt) and place them under `data/` as described in the
project structure below.

### 4. Train / evaluate

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
├── sources/               # research papers and utilities used in the project
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
reflected in the per-class scores — see `docs/graphs/` for class-distribution plots and
`docs/graphs/visualization/` for the confusion matrix and learned filters/activations.

## Documentation & Sources

- `docs/` — training graphs, mel-spectrograms per genre, logs, and the project report.
- `sources/` — research papers and other references that informed the approach.
