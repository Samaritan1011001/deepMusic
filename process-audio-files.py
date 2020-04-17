import os
# import time

import librosa
import pandas as pd

import utils
import numpy as np


def main():
    # root_folder = "/home/manojnb/deepMusic/"
    AUDIO_DIR = os.environ.get('AUDIO_DIR')
    # AUDIO_DIR = root_folder + 'fma_small'
    # start_time = time.time()

    tracks = pd.read_csv('fma_metadata/subset_small.csv', index_col=0)
    # tracks = pd.read_csv(root_folder + 'fma_metadata/subset_small.csv', index_col=0)

    tids = np.asarray(tracks.index)
    genres_top = np.array(tracks['track_genre_top'])
    print(genres_top.shape)
    # print(tids)
    res_pd = pd.DataFrame(columns=['signal','sr'])
    # for tid in tids:
    #     filename = utils.get_audio_path(AUDIO_DIR, tid)
    #     # print(filename)
    #     signal, sr = utils.FfmpegLoader().load(filename)
    #     # print(signal)
    #     signal = signal.astype(float)
    #     spect = compute_melgram(signal, sr)
    #     x_pd = pd.DataFrame({"signal": [spect],'sr':sr})
    #     res_pd = res_pd.append(x_pd, ignore_index=True)
        # print(spect.shape)
        # break
    # print(res_pd)
    # print(res_pd.loc[0]['signal'].shape)
    # res_pd.to_csv(root_folder + 'fma_metadata/processed_small.csv',index=False)
    # res_pd.to_csv('fma_metadata/processed_small.csv',index=False)

    print("Successful ", res_pd.size)

    res_pd = pd.read_csv('fma_metadata/processed_small.csv')
    # res_pd['tids'] = tids
    # res_pd['track_genre_top'] = genres_top
    print("res_pd",res_pd.columns)
    # res_pd.to_csv('fma_metadata/processed_small_tids_genres.csv')

    return
def split_datasets():
    # root_folder = "/home/manojnb/deepMusic/"
    AUDIO_DIR = os.environ.get('AUDIO_DIR')
    # AUDIO_DIR = root_folder + 'fma_small'
    # start_time = time.time()

    tracks = pd.read_csv('fma_metadata/subset_small.csv', index_col=0)
    train = tracks.loc[tracks['set_split'] == 'training']
    val = tracks.loc[tracks['set_split'] == 'validation']
    test = tracks.loc[tracks['set_split'] == 'test']

    res_pd = pd.read_csv('fma_metadata/processed_small_tids_genres.csv', index_col=0)
    train = res_pd.loc[res_pd['tids'].isin(list(train.index))]
    val = res_pd.loc[res_pd['tids'].isin(list(train.index))]
    test = res_pd.loc[res_pd['tids'].isin(list(train.index))]

    print(train.index.shape)



    return
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
                        n_fft=N_FFT, n_mels=N_MELS),
                ref=np.max)
    ret = ret[np.newaxis, np.newaxis, :]
    return ret
if __name__ == '__main__':
    main()
    # split_datasets()