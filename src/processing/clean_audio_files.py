# CLEANING CORRUPTED FILES

from src import utils
import librosa
import numpy as np
import time


root_folder = "/home/manojnb/deepMusic/"
AUDIO_DIR = root_folder + 'fma_medium'


tracks = utils.load(root_folder + 'fma_metadata/tracks.csv')
subset_small = tracks.index[tracks['set', 'subset'] == 'medium']
subset_small = tracks.loc[subset_small]

subset_small.columns = ['_'.join(col).strip() for col in subset_small.columns.values]
print(np.asarray(subset_small.index))
classes = list(np.unique(subset_small['track_genre_top']))
print(classes)
class_dist = subset_small.groupby(['track_genre_top'])['track_duration'].mean().dropna()
print(class_dist)
tids = np.asarray(subset_small.index)

# tids = [99134]
# tids = np.asarray(tids)
print(f'tids shape before -> {tids.shape}')
error_tids = []

start_time = time.time()
for tid in tids:
    filename = utils.get_audio_path(AUDIO_DIR, tid)
    try:
        x, sr = librosa.load(filename, sr=None, mono=True)
        # print(f'Loaded file {tid}\n')
    except Exception as e:
        print(f'Error loading -> {tid}\n')
        error_tids.append(tid)
        tids = np.delete(tids, np.argwhere(tids==tid))
        continue
print("--- %s seconds ---" % (time.time() - start_time))

print(error_tids)
subset_small.drop(error_tids,inplace = True)
subset_small.to_csv(root_folder + 'fma_metadata/cleaned_medium.csv')
# !cp subset_small.csv "/content/drive/My Drive/cs577- Deep learning/deepMusic/fma_metadata/"
# print(f'Deleted in metadata {subset_small.loc[99134]}')
print(f'tids shape after -> {tids.shape}')