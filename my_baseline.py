import time
import os

import IPython.display as ipd
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
import keras
from keras.layers import Activation, Dense, Conv1D, Conv2D, MaxPooling1D, Flatten, Reshape

from sklearn.utils import shuffle
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, LabelBinarizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier

import utils

def main():
    AUDIO_DIR = os.environ.get('AUDIO_DIR')

    tracks = utils.load('fma_metadata/tracks.csv')
    features = utils.load('fma_metadata/features.csv')
    echonest = utils.load('fma_metadata/echonest.csv')

    np.testing.assert_array_equal(features.index, tracks.index)
    assert echonest.index.isin(tracks.index).all()

    print(tracks.shape, features.shape, echonest.shape)


    subset = tracks.index[tracks['set', 'subset'] <= 'medium']

    assert subset.isin(tracks.index).all()
    assert subset.isin(features.index).all()

    features_all = features.join(echonest, how='inner').sort_index(axis=1)
    print('Not enough Echonest features: {}'.format(features_all.shape))

    tracks = tracks.loc[subset]
    features_all = features.loc[subset]

    train = tracks.index[tracks['set', 'split'] == 'training']
    val = tracks.index[tracks['set', 'split'] == 'validation']
    test = tracks.index[tracks['set', 'split'] == 'test']

    print('{} training examples, {} validation examples, {} testing examples'.format(*map(len, [train, val, test])))

    genres = list(LabelEncoder().fit(tracks['track', 'genre_top']).classes_)
    # genres = list(tracks['track', 'genre_top'].unique())
    print('Top genres ({}): {}'.format(len(genres), genres))
    genres = list(MultiLabelBinarizer().fit(tracks['track', 'genres_all']).classes_)
    print('All genres ({}): {}'.format(len(genres), genres))
    print(tracks.shape, features_all.shape)

    # Directory where mp3 are stored.
    AUDIO_DIR = os.environ.get('AUDIO_DIR')
    print(AUDIO_DIR)
    print(utils.get_audio_path(AUDIO_DIR, 2))

    labels_onehot = LabelBinarizer().fit_transform(tracks['track', 'genre_top'])
    labels_onehot = pd.DataFrame(labels_onehot, index=tracks.index)
    # Just be sure that everything is fine. Multiprocessing is tricky to debug.
    utils.FfmpegLoader().load(utils.get_audio_path(AUDIO_DIR, 2))
    # AUDIO_DIR = 'D:\Code\Desktop\IIT\Courses\Spring 2020\Deep Learning\FInal Project\github_project\music_analysis_fp\fma_small\000\000002.mp3'
    # utils.FfmpegLoader().load(r'D:\Code\Desktop\IIT\Courses\Spring 2020\Deep Learning\FInal Project\github_project\music_analysis_fp\fma_small\000\000002.mp3')
    SampleLoader = utils.build_sample_loader(AUDIO_DIR, labels_onehot, utils.FfmpegLoader())
    SampleLoader(train, batch_size=2).__next__()[0].shape

if __name__ == '__main__':
  main()

