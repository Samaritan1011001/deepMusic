
# !pip install python_speech_features
# !pip install python-dotenv

# !pip install keras --upgrade

import numpy as np
import pandas as pd
from python_speech_features import mfcc
import os
# import tensorflow as tf
from keras import layers
from keras import models
import keras
from keras import optimizers
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
# import sys
# sys.path.append('/content/drive/My Drive/cs577- Deep learning/deepMusic/')
import utils
from sklearn import preprocessing
import librosa

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)

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
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # # Store sample
            # signal, rate = self.loader.load(utils.get_audio_path(self.audio_dir, ID))
            # sample = signal[:rate]
            
            # # Shorten the sample to 10 secs
            # # ran_index = np.random.randint(0, signal.shape[0] - int(rate / 100))
            # # sample = signal[ran_index:ran_index + int(rate / 100)]
            # # print(f'shape of sample -> {sample.shape}')

            # normalized_X = preprocessing.normalize(mfcc(sample, rate, numcep=13, nfilt=26, nfft=1103).T)
            # temp = np.array(normalized_X)
            # rshaped_X = temp.reshape(temp.shape[0], temp.shape[1], 1)
            # X[i,] = rshaped_X

            # Librosa version mel spectrogram
            signal, sr = librosa.load(utils.get_audio_path(self.audio_dir, ID))
            # S = librosa.feature.melspectrogram(y=y[:int(sr/5)], sr=sr, n_mels=13,fmax=8000)
            spect = librosa.feature.melspectrogram(y=signal[:int(sr/5)], sr=sr, n_mels=13,fmax=8000)
            # spect = librosa.power_to_db(spect, ref=np.max)
            # print(spect.shape)
            temp = np.array(spect)
            rshaped_X = temp.reshape(temp.shape[0], temp.shape[1], 1)
            X[i,] = rshaped_X

            # Store class
            y[i] = self.labels.loc[ID].to_numpy()

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

# # # Train and Validate
# ff_loader = utils.FfmpegLoader()

# # METHOD 2 - DATAGENERATOR
# # Parameters
# params = {'dim': (13, 9),
#           'batch_size': 1,
#           'n_classes': 8,
#           'n_channels': 1,
#           'shuffle': True}
# training_generator = DataGenerator(train.index.values, labels_onehot,ff_loader, AUDIO_DIR, **params)
# x , y = training_generator.__getitem__(0)
# print(x.shape)
# print(y.shape)

# print(keras.__version__)
root_folder = "/home/manojnb/deepMusic/"
AUDIO_DIR = root_folder + 'fma_small'

# Read data
tracks = pd.read_csv(root_folder+'fma_metadata/subset_small.csv', index_col=0)
print(tracks['set_split'].shape)
train = tracks.loc[tracks['set_split'] == 'training']
val = tracks.loc[tracks['set_split'] == 'validation']
test = tracks.index[tracks['set_split'] == 'test']
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
print(f'train.index.values -> {len(val.index.values)}')

# Input shape is this because 13->numcep in mfcc function and 9 -> number of frames considered
# input_shape = (13, 9, 1)
input_shape = (13, 9, 1)

# Network Architecture
# tf.keras.backend.clear_session()

model = models.Sequential()

model.add(layers.Conv2D(16, (3, 3), activation='relu',
                        strides=(1, 1), padding='same',
                        input_shape=input_shape))
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        strides=(1, 1), padding='same',
                        ))
model.add(layers.Conv2D(64, (3, 3), activation='relu',
                        strides=(1, 1), padding='same',
                        ))
model.add(layers.Conv2D(128, (3, 3), activation='relu',
                        strides=(1, 1), padding='same',
                        ))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))
model.summary()

# optimizer = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
optimizer = optimizers.Adam(lr=0.01)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# # Train and Validate
ff_loader = utils.FfmpegLoader()

# METHOD 2 - DATAGENERATOR
# Parameters
params = {'dim': (13, 9),
          'batch_size': 10,
          'n_classes': 8,
          'n_channels': 1,
          'shuffle': True}

# Generators
training_generator = DataGenerator(train.index.values, labels_onehot,ff_loader,AUDIO_DIR, **params)
validation_generator = DataGenerator(val.index.values, labels_onehot,ff_loader,AUDIO_DIR, **params)

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=1,
                    use_multiprocessing=True, workers=1)