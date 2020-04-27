# CLEANING CORRUPTED FILES

import time

import librosa
import numpy as np

from sources import utils

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
subset_small = tracks.index[tracks['set', 'subset'] == subset_config['subset']]
subset_small = tracks.loc[subset_small]

subset_small.columns = ['_'.join(col).strip() for col in subset_small.columns.values]
print(np.asarray(subset_small.index))
classes = list(np.unique(subset_small['track_genre_top']))
print(classes)
class_dist = subset_small.groupby(['track_genre_top'])['track_duration'].mean().dropna()
print(class_dist)
tids = np.asarray(subset_small.index)

print(f'tids shape before -> {tids.shape}')
error_tids = []

start_time = time.time()
for tid in tids:
    filename = utils.get_audio_path(folder_cofig['AUDIO_DIR'], tid)
    try:
        x, sr = librosa.load(filename, sr=None, mono=True)
    except Exception as e:
        print(f'Error loading -> {tid}\n')
        error_tids.append(tid)
        tids = np.delete(tids, np.argwhere(tids == tid))
        continue
print("--- %s seconds ---" % (time.time() - start_time))

print(error_tids)
subset_small.drop(error_tids, inplace=True)
# UNCOMMENT TO SAVE
# subset_small.to_csv(root_folder + 'data/fma_metadata/cleaned_' + subset_config['subset'] + '.csv')
print(f'tids shape after -> {tids.shape}')
