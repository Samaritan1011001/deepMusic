import numpy as np
import pandas as pd
from python_speech_features import mfcc
import os
from keras import layers
from keras import models
import keras
from keras import optimizers
from keras import regularizers
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import utils
from sklearn import preprocessing
import librosa
import matplotlib.pyplot as plt
import librosa.display

import librosa
import numpy as np
from math import floor

from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Reshape, Permute, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import GRU
from keras.utils.data_utils import get_file


import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus


def plot_mel_spect(loader,audio_dir,ID):
    signal, sr = loader.load(utils.get_audio_path(audio_dir, ID))
    signal = signal.astype(float)
    # Shorten the sample to 10 secs
    ran_index = np.random.randint(0, signal.shape[0] - int(sr / 1))
    sample = signal[ran_index:ran_index + int(sr / 1)]
    spect = librosa.feature.melspectrogram(y=sample, sr=sr, n_mels=64,fmax=8000)
    
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(spect.T, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Test Melspectogram')
    plt.show()

# root_folder = "/home/manojnb/deepMusic/"
# AUDIO_DIR = root_folder + 'fma_small'

root_folder = "D:\Code\Desktop\IIT\Courses\Spring 2020\Deep Learning\FInal Project\github_project\music_analysis_fp\\"
AUDIO_DIR = os.environ.get('AUDIO_DIR')


# Read data
tracks = pd.read_csv(root_folder+'fma_metadata/subset_small.csv', index_col=0)
print(tracks['set_split'].shape)
train = tracks.loc[tracks['set_split'] == 'training']
val = tracks.loc[tracks['set_split'] == 'validation']
test = tracks.loc[tracks['set_split'] == 'test']
print(f'train -> {train.columns}')

# Check which genres are present
genres = list(LabelEncoder().fit(train['track_genre_top']).classes_)
print('Top genres ({}): {}'.format(len(genres), genres))

# one hot encode y
# labels_onehot = LabelBinarizer().fit_transform(tracks['track_genre_top'])
# labels_onehot = pd.DataFrame(labels_onehot, index=tracks.index)
# print(f'labels_onehot -> {labels_onehot.head()}')

labels_onehot = np.asarray(LabelEncoder().fit_transform(tracks['track_genre_top']))
labels_onehot = pd.DataFrame(labels_onehot, index=tracks.index)
print(f'labels_onehot -> {labels_onehot.head()}')
# print(f'label type {labels_onehot.shape}')

# For local training purposes
train = train.head(100)
val = val.head(100)
test = test.head(100)
print(f'test.index.values -> {len(test.index.values)}')


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
    DURA = 29.12  # to make it 1366 frame..

    # src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA*SR)
    # print(src[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)])


    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)]
    logam = librosa.amplitude_to_db
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                        n_fft=N_FFT, n_mels=N_MELS)**2,
                ref=np.max)
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
            signal, sr = self.loader.load(utils.get_audio_path(self.audio_dir, ID))
            signal = signal.astype(float)
            spect = compute_melgram(signal, sr)
            # # Shorten the sample to 10 secs
            # ran_index = np.random.randint(0, signal.shape[0] - int(sr / 1))
            # sample = signal[ran_index:ran_index + int(sr / 1)]
            # # print(sr)
            # spect = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=96,n_fft=512,hop_length=256)
            
            # plt.figure(figsize=(10, 5))
            # librosa.display.specshow(spect.T, y_axis='mel', x_axis='time')
            # plt.colorbar(format='%+2.0f dB')
            # plt.title('Test Melspectogram')
            # plt.show()
            X[i,] = spect
            # temp = np.array(spect)
            # rshaped_X = temp.reshape(temp.shape[0], temp.shape[1], 1)
            # X[i,] = rshaped_X

            # X[i,] = np.array(spect)

            # Store class
            y[i] = self.labels.loc[ID].to_numpy()

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

# # Train and Validate
ff_loader = utils.FfmpegLoader()

# Plot mel spect for audio file with ID 2
# plot_mel_spect(ff_loader,AUDIO_DIR,2)

# METHOD 2 - DATAGENERATOR
# Parameters
params = {'dim': (96,1366),
          'batch_size': 10,
          'n_classes': 8,
          'n_channels': 1,
          'shuffle': True}
training_generator = DataGenerator(train.index.values, labels_onehot,ff_loader, AUDIO_DIR, **params)
x , y = training_generator.__getitem__(0)
print(x.shape)
# print(train.loc[ID]['track_genre_top'])
print(y)


def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False


def MusicTaggerCRNN(weights='msd', input_tensor=None):
    '''Instantiate the MusicTaggerCRNN architecture,
    optionally loading weights pre-trained
    on Million Song Dataset. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    For preparing mel-spectrogram input, see
    `audio_conv_utils.py` in [applications](https://github.com/fchollet/keras/tree/master/keras/applications).
    You will need to install [Librosa](http://librosa.github.io/librosa/)
    to use it.

    # Arguments
        weights: one of `None` (random initialization)
            or "msd" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
    # Returns
        A Keras model instance.
    '''

    K.set_image_data_format('channels_first')
    if weights not in {'msd', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `msd` '
                         '(pre-training on Million Song Dataset).')

    # Determine proper input shape
    if K.image_data_format() == 'channels_first':
        input_shape = (1, 96, 1366)
    else:
        input_shape = (96, 1366, 1)

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        melgram_input = Input(shape=input_tensor)

    # Determine input axis
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    # Input block
    x = ZeroPadding2D(padding=(0, 37))(melgram_input)
    x = BatchNormalization(axis=time_axis, name='bn_0_freq')(x)

    # Conv block 1
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv1', trainable=False)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1', trainable=False)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1', trainable=False)(x)
    x = Dropout(0.1, name='dropout1', trainable=False)(x)

    # Conv block 2
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv2', trainable=False)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2', trainable=False)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2', trainable=False)(x)
    x = Dropout(0.1, name='dropout2', trainable=False)(x)

    # Conv block 3
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv3', trainable=False)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3', trainable=False)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3', trainable=False)(x)
    x = Dropout(0.1, name='dropout3', trainable=False)(x)

    # Conv block 4
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv4', trainable=False)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4', trainable=False)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4', trainable=False)(x)
    x = Dropout(0.1, name='dropout4', trainable=False)(x)

    # reshaping
    if K.image_data_format() == 'channels_first':
        x = Permute((3, 1, 2))(x)
    x = Reshape((15, 128))(x)

    # GRU block 1, 2, output
    x = GRU(32, return_sequences=True, name='gru1')(x)
    x = GRU(32, return_sequences=False, name='gru2')(x)
    x = Dropout(0.3, name='final_drop')(x)

    if weights is None:
        # Create model
        x = Dense(8, activation='sigmoid', name='output')(x)
        model = Model(melgram_input, x)
        # model.summary()
        return model
    else:
        # Load input
        x = Dense(10, activation='sigmoid', name='output')(x)
        if K.image_data_format() == 'channels_last':
            raise RuntimeError("Please set image_dim_ordering == 'th'."
                               "You can set it at ~/.keras/keras.json")
        # Create model
        initial_model = Model(melgram_input, x)
        initial_model.load_weights(root_folder + 'crnn_net_gru_adam_ours_epoch_40.h5')

        # Eliminate last layer
        pop_layer(initial_model)
        pop_layer(initial_model)
        pop_layer(initial_model)
        pop_layer(initial_model)
        pop_layer(initial_model)
        pop_layer(initial_model)
        pop_layer(initial_model)
        pop_layer(initial_model)
        pop_layer(initial_model)
        pop_layer(initial_model)
        pop_layer(initial_model)
        pop_layer(initial_model)
        pop_layer(initial_model)
        pop_layer(initial_model)
        pop_layer(initial_model)
        pop_layer(initial_model)
        print(initial_model.layers)

        # Add new Dense layer
        # last = initial_model.get_layer('final_drop')
        # preds = (Dense(8, activation='softmax', name='preds'))(last.output)
        # model = Model(initial_model.input, last.output)

        # TO extend this model use this,
        model = Model(initial_model.input, initial_model.get_layer('dropout2').output)

        return model

def DeepMusicCRNN(input_tensor=None):
    '''
    # Arguments
        weights: one of `None` (random initialization)
            or "msd" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
    # Returns
        A Keras model instance.
    '''

    K.set_image_data_format('channels_first')


    # Determine input axis
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    input_x = Input(shape=input_tensor)
    # Input block
    # x = ZeroPadding2D(padding=(0, 37),input_shape = input_tensor)(input_x)
    x = BatchNormalization(axis=time_axis, name='bn_0b_freq')(input_x)

    # # Conv block 1
    x = Convolution2D(256, 3, 3, border_mode='same', name='conv1b', trainable=True)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1b', trainable=True)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(3, 3), name='pool1b', trainable=True)(x)
    x = Dropout(0.1, name='dropout1b', trainable=True)(x)

    # # Conv block 2
    x = Convolution2D(256, 3, 3, border_mode='same', name='conv2a', trainable=True)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2a', trainable=True)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2a', trainable=True)(x)
    x = Dropout(0.1, name='dropout2', trainable=True)(x)

    # # Conv block 3
    # x = Convolution2D(128, 3, 3, border_mode='same', name='conv3', trainable=False)(x)
    # x = BatchNormalization(axis=channel_axis, mode=0, name='bn3', trainable=False)(x)
    # x = ELU()(x)
    # x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3', trainable=False)(x)
    # x = Dropout(0.1, name='dropout3', trainable=False)(x)

    # # Conv block 4
    # x = Convolution2D(128, 3, 3, border_mode='same', name='conv4', trainable=False)(x)
    # x = BatchNormalization(axis=channel_axis, mode=0, name='bn4', trainable=False)(x)
    # x = ELU()(x)
    # x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4', trainable=False)(x)
    # x = Dropout(0.1, name='dropout4', trainable=False)(x)

    # reshaping
    # if K.image_data_format() == 'channels_first':
    #     x = Permute((3, 1, 2))(x)
    # x = Reshape((15, 256))(x)

    # # # GRU block 1, 2, output
    # x = GRU(32, return_sequences=True, name='gru1')(x)
    # x = GRU(32, return_sequences=False, name='gru2')(x)
    # x = Dropout(0.3, name='final_drop')(x)

    # Create model
    x = Flatten()(x)
    x = Dense(16, activation='relu', name='hidden')(x)
    x = Dense(8, activation='softmax', name='output')(x)
    model = Model(input_x, x)
    print(x.shape)
    # model.summary()
    return model

def plot_epocs_graph(history_dict):
    loss_vals = history_dict['loss']
    val_loss_vals = history_dict['val_loss']
    epochs = range(1, len(history_dict['acc']) + 1)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_vals, 'g', label='Training Loss')
    plt.plot(epochs, val_loss_vals, 'b', label='Validation Loss')
    plt.title("Training and validation loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    acc_vals = history_dict['acc']
    val_acc_vals = history_dict['val_acc']
    plt.plot(epochs, acc_vals, 'g', label='Training accuracy')
    plt.plot(epochs, val_acc_vals, 'b', label='Validation accuracy')
    plt.title("Training and validation Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.savefig("../doc/graphs/val_loss_acc_epochs_before1_task2.png")
    # plt.savefig("../doc/graphs/val_loss_acc_epochs_after1_task2.png")
    plt.show()
    return

# cnn
# model = gen_model()

# CRNN
# input_shape = (1,96,1366)
# model_input = layers.Input(input_shape, name='input')
# model = conv_recurrent_model_build(model_input,input_shape)

# CNN by mdeff
# input_shape = (1,96,1366)
# model = getMdeff_model(input_shape)

keras.backend.clear_session()
# Another trial CRNN
model = models.Sequential()
model.add(MusicTaggerCRNN(weights='msd'))
model.add(DeepMusicCRNN(input_tensor = (128,16,240)))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
model.summary()




# # Generators
training_generator = DataGenerator(train.index.values, labels_onehot,ff_loader,AUDIO_DIR, **params)
validation_generator = DataGenerator(val.index.values, labels_onehot,ff_loader,AUDIO_DIR, **params)
test_generator = DataGenerator(test.index.values, labels_onehot,ff_loader,AUDIO_DIR, **params)

# # Train model on dataset
train_val_history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=5,
                    use_multiprocessing=True, workers=1,
                    # steps_per_epoch = 20,

                    )
history_dict = train_val_history.history
plot_epocs_graph(history_dict=history_dict)
