import time
import numpy as np
import pandas as pd
import keras

from sklearn.preprocessing import LabelEncoder
from src import utils
import librosa
import matplotlib.pyplot as plt
import librosa.display
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)


def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    # global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]


tfback._get_available_gpus = _get_available_gpus

gen_params = {
    'dim': (96,1366),
    # 'dim': (96, 469),
    'batch_size': 10,
    'n_classes': 4,
    'n_channels': 1,
    'shuffle': True}

root_folder = "/home/manojnb/deepMusic/"
config = {
    'audio_dir': root_folder + "fma_medium",
    'tracks': root_folder + 'fma_metadata/cleaned_medium.csv',
    'generator_params': gen_params,
    'audio_loader': utils.FfmpegLoader()

}


# config['generator_params']['dim']

def plot_mel_spect(spect):
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(spect.T, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Test Melspectogram')
    plt.show()


# Read data
tracks = pd.read_csv(config['tracks'], index_col=0)
tracks = tracks[tracks.track_genre_top.isin(['Electronic', 'Folk', 'Rock', 'Hip-Hop'])]
# ['Electronic', 'Folk', 'Rock']
print(tracks['set_split'].shape)
train = tracks.loc[tracks['set_split'] == 'training']
val = tracks.loc[tracks['set_split'] == 'validation']
test = tracks.loc[tracks['set_split'] == 'test']
print(f'train -> {train.columns}')

# Check which genres are present
genres = list(LabelEncoder().fit(train['track_genre_top']).classes_)
print('Top genres ({}): {}'.format(len(genres), genres))
le = LabelEncoder()
labels_encoded = np.asarray(le.fit_transform(tracks['track_genre_top']))
labels_encoded = pd.DataFrame(labels_encoded, index=tracks.index)

y_integers = labels_encoded.to_numpy().flatten()
# print(f'labels_encoded -> {y_integers}')
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))
# print(f'd_class_weights -> {d_class_weights}')

# For local training purposes
# train = train.head(10)
# val = val.head(10)
# test = test.head(10)
#
# print(f'train y -> {train["track_genre_top"].unique()}')
# print(f'val y -> {val["track_genre_top"].unique()}')
# print(f'test y -> {test["track_genre_top"].unique()}')
print(f'train count -> {train.shape}')
print(f'val count -> {val.shape}')
print(f'test count -> {test.shape}')


def compute_melgram(src, sr):
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
    # print(f'n_sample_fit -> {n_sample_fit}')
    # print(f'n_sample -> {n_sample}')
    # print(src[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)])

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[int((n_sample - n_sample_fit) / 2):int((n_sample + n_sample_fit) / 2)]
    logam = librosa.amplitude_to_db
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                        n_fft=N_FFT, n_mels=N_MELS) ** 2,
                ref=np.max)
    # plot_mel_spect(ret)
    ret = ret[np.newaxis, np.newaxis, :]
    return ret


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, labels, loader, audio_dir, batch_size=32, dim=(32, 32, 32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.loader = loader
        self.audio_dir = audio_dir

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # print(list_IDs_temp)

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.n_channels, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # # Store sample
            # signal, rate = self.loader.load(utils.get_audio_path(self.audio_dir, ID))
            # sample = signal[:rate]

            # # Shorten the sample to 10 secs
            # ran_index = np.random.randint(0, signal.shape[0] - int(rate / 10))
            # sample = signal[ran_index:ran_index + int(rate / 10)]
            # # print(f'shape of sample -> {sample.shape}')
            # normalized_X = preprocessing.normalize(mfcc(sample, rate, numcep=13, nfilt=26, nfft=1103).T)
            # temp = np.array(normalized_X)
            # rshaped_X = temp.reshape(temp.shape[0], temp.shape[1], 1)
            # X[i,] = rshaped_X

            # Librosa version mel spectrogram
            signal, sr = librosa.load(utils.get_audio_path(self.audio_dir, ID))
            signal = signal.astype(float)
            spect = compute_melgram(signal, sr)
            X[i,] = spect

            # temp = np.array(spect)
            # rshaped_X = temp.reshape(temp.shape[0], temp.shape[1], 1)
            # X[i,] = rshaped_X
            # X[i,] = np.array(spect)

            # Store class
            y[i] = self.labels.loc[ID].to_numpy()

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []
training_generator = DataGenerator(train.index.values, labels_encoded, config['audio_loader'], config['audio_dir'],
                                   **config['generator_params'])
validation_generator = DataGenerator(val.index.values, labels_encoded, config['audio_loader'], config['audio_dir'],
                                     **config['generator_params'])
test_generator = DataGenerator(test.index.values, labels_encoded, config['audio_loader'], config['audio_dir'],
                               **config['generator_params'])

start_time = time.time()
train_batches = len(train.index.values) // config['generator_params']['batch_size']
print(f'train_batches -> {train_batches}')
for j in range(train_batches):
    x, y = training_generator.__getitem__(j)
    x_train.append(x)
    y_train.append(y)

test_val_batches = len(test.index.values) // config['generator_params']['batch_size']
print(f'test_val_batches -> {test_val_batches}')

for i in range(test_val_batches):
    x_v, y_v = validation_generator.__getitem__(i)
    x_val.append(x_v)
    y_val.append(y_v)

    x_t, y_t = test_generator.__getitem__(i)
    x_test.append(x_t)
    y_test.append(y_t)

# print(f'x_train -> {np.asarray(x_train)}')
np.save(root_folder + "npy_files/medium/x_train.npy", x_train)
np.save(root_folder + "npy_files/medium/y_train.npy", y_train)

np.save(root_folder + "npy_files/medium/x_val.npy", x_val)
np.save(root_folder + "npy_files/medium/y_val.npy", y_val)

np.save(root_folder + "npy_files/medium/x_test.npy", x_test)
np.save(root_folder + "npy_files/medium/y_test.npy", y_test)

# x_train = list(np.load(root_folder + "npy_files/x_train.npy"))
print(f'x_train -> {len(x_train)}')
print(f'x_val -> {len(x_val)}')
print(f'x_test -> {len(x_test)}')

print("--- %s seconds toload data ---" % (time.time() - start_time))
