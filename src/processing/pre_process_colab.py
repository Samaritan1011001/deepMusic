import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from sources import utils
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from python_speech_features import mfcc, logfbank
import librosa.display


def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=4, sharex=False,
                             sharey=True, figsize=(20, 5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(4):
            axes[x, y].set_title(list(signals.keys())[i])
            axes[x, y].plot(list(signals.values())[i])
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=4, sharex=False,
                             sharey=True, figsize=(20, 4))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(4):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x, y].set_title(list(fft.keys())[i])
            axes[x, y].plot(freq, Y)
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=4, sharex=False,
                             sharey=True, figsize=(20, 4))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(4):
            axes[x, y].set_title(list(fbank.keys())[i])
            axes[x, y].imshow(list(fbank.values())[i],
                              cmap='hot', interpolation='nearest')
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=4, sharex=False,
                             sharey=True, figsize=(20, 4))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(4):
            axes[x, y].set_title(list(mfccs.keys())[i])
            axes[x, y].imshow(list(mfccs.values())[i],
                              cmap='hot', interpolation='nearest')
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


def plot_mel_spect(spect, key):
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(spect.T, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Test Melspectogram')
    key = key.replace(" / s", "")
    plt.savefig(root_folder + f'docs/graphs/{subset_config["subset"]}/mel_spect_{key}.png')


root_folder = "D:\Code\Desktop\IIT\Courses\Spring 2020\Deep Learning\FInal Project\github_project\music_analysis_fp\\"
# root_folder = "/home/manojnb/deepMusic/"


# subset_config = {
#     "audio_dir" : "fma_medium",
#     "subset" : "medium"
# }
subset_config = {
    "audio_dir": "fma_small",
    "subset": "small"
}
folder_cofig = {
    "AUDIO_DIR": root_folder + "data/" + subset_config['audio_dir']
}

tracks = utils.load(root_folder + 'data/fma_metadata/tracks.csv')
features = utils.load(root_folder + 'data/fma_metadata/features.csv')
echonest = utils.load(root_folder + 'data/fma_metadata/echonest.csv')

np.testing.assert_array_equal(features.index, tracks.index)
assert echonest.index.isin(tracks.index).all()

tracks.shape, features.shape, echonest.shape

subset = tracks.index[tracks['set', 'subset'] == subset_config['subset']]

assert subset.isin(tracks.index).all()
assert subset.isin(features.index).all()

features_all = features.join(echonest, how='inner').sort_index(axis=1)
print('Not enough Echonest features: {}'.format(features_all.shape))

tracks = tracks.loc[subset]
features_all = features.loc[subset]

tracks.shape, features_all.shape

train = tracks[tracks['set', 'split'] == 'training']
val = tracks[tracks['set', 'split'] == 'validation']
test = tracks[tracks['set', 'split'] == 'test']

print('{} training examples, {} validation examples, {} testing examples'.format(*map(len, [train, val, test])))

# genres = list(LabelEncoder().fit(tracks['track', 'genre_top']).classes_)
genres = list(tracks['track', 'genre_top'].unique())
print('Top genres ({}): {}'.format(len(genres), genres))

all_tracks = tracks.copy()
all_tracks.columns = ['_'.join(col).strip() for col in all_tracks.columns.values]
print(all_tracks.columns)

classes = list(np.unique(all_tracks['track_genre_top']))
print(classes)
class_dist = all_tracks.groupby(['track_genre_top'])['track_duration'].mean().dropna()
print(class_dist)

fig, ax = plt.subplots()
ax.set_title('Class distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct="%1.1f%%", shadow=False, startangle=90)
ax.axis('equal')
plt.savefig(root_folder + "docs/graphs/" + subset_config['subset'] + "/class_dist.png")
plt.show()
all_tracks.reset_index(inplace=True)


def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1 / rate)
    Y = abs(np.fft.rfft(y) / n)
    return (Y, freq)


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate / 10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


def compute_melgram(src, sr, c):
    ''' Compute a mel-spectrogram and returns it in a shape of (1,1,96,1366), where
    96 == #mel-bins and 1366 == #time frame

    parameters
    ----------
    audio_path: path for the audio file.
                Any format supported by audioread will work.
    More info: http://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load

    '''

    # mel-spectrogram parameters
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    # DURA = 10
    DURA = 29.12  # to make it 1366 frame..

    # src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA * SR)
    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[int((n_sample - n_sample_fit) / 2):int((n_sample + n_sample_fit) / 2)]
    logam = librosa.amplitude_to_db
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                        n_fft=N_FFT, n_mels=N_MELS) ** 2,
                ref=np.max)
    plot_mel_spect(ret, c)
    ret = ret[np.newaxis, np.newaxis, :]
    return spects


# Directory where mp3 are stored.
# AUDIO_DIR = os.environ.get('AUDIO_DIR')
print(folder_cofig['AUDIO_DIR'])

signals = {}
spects = {}
fft = {}
fbank = {}
mfccs = {}

for c in classes:
    mp3_file = all_tracks[all_tracks.track_genre_top == c].iloc[0, 0]
    signal, rate = librosa.load(utils.get_audio_path(folder_cofig['AUDIO_DIR'], mp3_file))
    s = signal.astype(float)
    compute_melgram(s, rate, c)
    signals[c] = signal
    fft[c] = calc_fft(signal, rate)
    bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
    fbank[c] = bank
    mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
    mfccs[c] = mel

print(rate)

plot_signals(signals)
plt.savefig(root_folder + "docs/graphs/" + subset_config['subset'] + "/signals.png")
plt.show()

plot_fft(fft)
plt.savefig(root_folder + "docs/graphs/" + subset_config['subset'] + "/fft.png")
plt.show()

plot_fbank(fbank)
plt.savefig(root_folder + "docs/graphs/" + subset_config['subset'] + "/fbank.png")
plt.show()

plot_mfccs(mfccs)
plt.savefig(root_folder + "docs/graphs/" + subset_config['subset'] + "/mfcc.png")
plt.show()

for key, spect in spects.items():
    plot_mel_spect(spect, key)
