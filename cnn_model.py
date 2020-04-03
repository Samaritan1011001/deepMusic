
import numpy as np
import pandas as pd

from python_speech_features import mfcc, logfbank
import os

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, LabelBinarizer, StandardScaler

import utils
from sklearn import preprocessing
def build_sample_loader(audio_dir, Y, loader):

    class SampleLoader:
        def __init__(self, tids, batch_size=4):
#             self.lock1 = multiprocessing.Lock()
#             self.lock2 = multiprocessing.Lock()
            self.batch_foremost = 0
#             self.batch_rearmost = sharedctypes.RawValue(ctypes.c_int, -1)
#             self.condition = multiprocessing.Condition(lock=self.lock2)
#             data = sharedctypes.RawArray(ctypes.c_int, tids.data)
#             self.tids = np.ctypeslib.as_array(data)
            self.tids = np.asarray(tids.data)
# #             print(f'self.tids-> {type(self.tids)}')
            self.batch_size = batch_size
            self.loader = loader
            self.X = []
            self.Y = []
        def __iter__(self):
            return self
        def __next__(self):
#             with self.lock1:
            if self.batch_foremost == 0:
                np.random.shuffle(self.tids)
            batch_current = self.batch_foremost
            if self.batch_foremost + self.batch_size < self.tids.size:
                batch_size = self.batch_size
                self.batch_foremost += self.batch_size
            else:
                batch_size = self.tids.size - self.batch_foremost
                self.batch_foremost = 0
            tids = np.array(self.tids[batch_current:batch_current+batch_size])
# #             print(tids)
            for i, tid in enumerate(tids):
#                 self.X[i] = self.loader.load(get_audio_path(audio_dir, tid))
                signal, rate = self.loader.load(utils.get_audio_path(audio_dir, tid))
# #                 print(f'signal -> {signal.shape}')
# #                 print(f'rate -> {rate}')
# #                 print(f'signal rate -> {signal[:rate].shape}')
# #                 print(f'self.x -> {self.X[i].shape}')
# #                 print(f'mfcc -> {mfcc(signal[:rate],rate, numcep=13, nfilt=26, nfft=1103).T.shape}')
                ran_index = np.random.randint(0,signal.shape[0]-int(rate/10))
                sample = signal[ran_index:ran_index+int(rate/10)]
                normalized_X = preprocessing.normalize(mfcc(sample,22050,numcep=13, nfilt=26, nfft=1103).T)
# #                 print(f'norm x shape {normalized_X.shape}')
                self.X.append(normalized_X)
                self.Y.append(Y.loc[tid])
            temp = np.array(self.X[:batch_size])
# #             print(f'temp x shape {temp.shape}')
            rshaped_X = temp.reshape(temp.shape[0],temp.shape[1],temp.shape[2],1)
            return rshaped_X , np.array(self.Y[:batch_size])
    return SampleLoader
AUDIO_DIR = os.environ.get('AUDIO_DIR')
tracks = utils.load('fma_metadata/tracks.csv')
features = utils.load('fma_metadata/features.csv')
echonest = utils.load('fma_metadata/echonest.csv')
np.testing.assert_array_equal(features.index, tracks.index)
assert echonest.index.isin(tracks.index).all()
# print(tracks.shape, features.shape, echonest.shape)


subset = tracks.index[tracks['set', 'subset'] == 'small']
assert subset.isin(tracks.index).all()
assert subset.isin(features.index).all()
features_all = features.join(echonest, how='inner').sort_index(axis=1)
# print('Not enough Echonest features: {}'.format(features_all.shape))
tracks = tracks.loc[subset]
features_all = features.loc[subset]
# print(tracks.shape, features_all.shape)


train = tracks.index[tracks['set', 'split'] == 'training']
val = tracks.index[tracks['set', 'split'] == 'validation']
test = tracks.index[tracks['set', 'split'] == 'test']

# print('{} training examples, {} validation examples, {} testing examples'.format(*map(len, [train, val, test])))

genres = list(LabelEncoder().fit(tracks['track', 'genre_top']).classes_)
#genres = list(tracks['track', 'genre_top'].unique())
# print('Top genres ({}): {}'.format(len(genres), genres))
genres = list(MultiLabelBinarizer().fit(tracks['track', 'genres_all']).classes_)
# print('All genres ({}): {}'.format(len(genres), genres))
labels_onehot = LabelBinarizer().fit_transform(tracks['track', 'genre_top'])
labels_onehot = pd.DataFrame(labels_onehot, index=tracks.index)
loader = utils.FfmpegLoader()
SampleLoader = build_sample_loader(AUDIO_DIR, labels_onehot, loader)
# print('Dimensionality: {}'.format(loader.shape))
sample_X_shape = SampleLoader(train, batch_size=10).__next__()[0].shape
# print(sample_X_shape)


input_shape = (sample_X_shape[1],sample_X_shape[2],1)
# print(input_shape)
# keras.backend.clear_session()

model = keras.models.Sequential()

model.add(layers.Conv2D(16,(3,3),activation = 'relu',
                        strides=(1,1), padding='same',
                        input_shape=input_shape))
model.add(layers.Conv2D(32,(3,3),activation = 'relu',
                        strides=(1,1), padding='same',
                        ))
model.add(layers.Conv2D(64,(3,3),activation = 'relu',
                        strides=(1,1), padding='same',
                        ))
model.add(layers.Conv2D(128,(3,3),activation = 'relu',
                        strides=(1,1), padding='same',
                        ))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))
model.summary()

optimizer = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])



model.fit_generator(SampleLoader(train, batch_size=10), steps_per_epoch=len(train) // 10, epochs=20)
loss = model.evaluate_generator(SampleLoader(val, batch_size=10), val.size)
# loss = model.evaluate_generator(SampleLoader(test, batch_size=10), test.size)

# print(loss)