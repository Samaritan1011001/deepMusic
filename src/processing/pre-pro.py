from python_speech_features import logfbank, mfcc

from src import utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=4, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(4):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=4, sharex=False,
                             sharey=True, figsize=(20,4))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(4):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=4, sharex=False,
                             sharey=True, figsize=(20,4))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(4):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=4, sharex=False,
                             sharey=True, figsize=(20,4))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(4):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def main():
    # AUDIO_DIR = os.environ.get('AUDIO_DIR')
    project_root = "/home/manojnb/deepMusic/"
    tracks = utils.load(project_root + 'fma_metadata/tracks.csv')

    subset = tracks.index[tracks['set', 'subset'] == 'small']

    assert subset.isin(tracks.index).all()
    tracks = tracks.loc[subset]

    train = tracks.index[tracks['set', 'split'] == 'training']
    val = tracks.index[tracks['set', 'split'] == 'validation']
    test = tracks.index[tracks['set', 'split'] == 'test']

    print('{} training examples, {} validation examples, {} testing examples'.format(*map(len, [train, val, test])))

    # genres = list(LabelEncoder().fit(tracks['track', 'genre_top']).classes_)
    genres = list(tracks['track', 'genre_top'].unique())
    print('Top genres ({}): {}'.format(len(genres), genres))

    all_tracks = tracks.copy()
    all_tracks.columns = ['_'.join(col).strip() for col in all_tracks.columns]
    print(all_tracks.columns)


    print(all_tracks.shape)


    classes = list(np.unique(all_tracks['track_genre_top']))
    print(classes)
    class_dist = all_tracks.groupby(['track_genre_top'])['track_duration'].mean().dropna()
    print(class_dist)


    fig, ax = plt.subplots()
    ax.set_title('Class distribution', y=1.08)
    ax.pie(class_dist,labels=class_dist.index,autopct="%1.1f%%",shadow=False,startangle=90)
    ax.axis('equal')
    plt.savefig(project_root + "graphs/class_dist.png")
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
    #
    # Directory where mp3 are stored.
    AUDIO_DIR = project_root + "fma_small"
    print(AUDIO_DIR)

    signals = {}
    fft = {}
    fbank = {}
    mfccs = {}
    #
    for c in classes:
        mp3_file = all_tracks[all_tracks.track_genre_top == c].iloc[0, 0]
        signal, rate = utils.FfmpegLoader().load(utils.get_audio_path(AUDIO_DIR, mp3_file))
        #     mask = envelope(signal, rate,0.5)
        #     signal = signal[mask]
        signals[c] = signal
        fft[c] = calc_fft(signal, rate)

        bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
        fbank[c] = bank
        mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
        mfccs[c] = mel

    print("Successful")
    #
    plot_signals(signals)
    plt.savefig(project_root + "graphs/signals.png")
    # plt.show()

    plot_fft(fft)
    plt.savefig(project_root + "graphs/fft.png")
    # plt.show()

    plot_fbank(fbank)
    plt.savefig(project_root + "graphs/fbank.png")
    # plt.show()

    plot_mfccs(mfccs)
    plt.savefig(project_root + "graphs/mfcc.png")
    # plt.show()

if __name__ == '__main__':
    main()