
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
sys.path.append('/content/drive/My Drive/cs577- Deep learning/deepMusic')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

root_folder = "/home/manojnb/deepMusic/"
config = {
    'audio_dir': root_folder + "fma_medium",
    'tracks': root_folder + 'fma_metadata/cleaned_medium.csv',
}
# Read data
tracks = pd.read_csv(config['tracks'], index_col=0)
tracks = tracks[tracks.track_genre_top.isin(['Electronic', 'Folk', 'Rock', 'Hip-Hop'])]
# ['Electronic', 'Folk', 'Rock']
print(tracks['set_split'].shape)
train = tracks.loc[tracks['set_split'] == 'training']
val = tracks.loc[tracks['set_split'] == 'validation']
test = tracks.loc[tracks['set_split'] == 'test']

print('{} training examples, {} validation examples, {} testing examples'.format(*map(len, [train, val, test])))

# genres = list(LabelEncoder().fit(tracks['track', 'genre_top']).classes_)
genres = list(tracks['track_genre_top'].unique())
print('Top genres ({}): {}'.format(len(genres), genres))

all_tracks = tracks.copy()
# all_tracks.columns = ['_'.join(col).strip() for col in all_tracks.columns.values]
# print(all_tracks.columns)

all_classes = list(np.unique(all_tracks['track_genre_top']))
print(all_classes)
all_class_dist = all_tracks.groupby(['track_genre_top'])['track_duration'].mean().dropna()
print(all_class_dist)

train_classes = list(np.unique(train['track_genre_top']))
print(train_classes)
train_class_dist = train.groupby(['track_genre_top'])['track_duration'].mean().dropna()
print(train_class_dist)

val_classes = list(np.unique(val['track_genre_top']))
# print(classes)
val_class_dist = val.groupby(['track_genre_top'])['track_duration'].mean().dropna()
# print(class_dist)

test_classes = list(np.unique(test['track_genre_top']))
# print(classes)
test_class_dist = test.groupby(['track_genre_top'])['track_duration'].mean().dropna()
# print(class_dist)

fig, ax = plt.subplots()
ax.set_title('Class distribution', y=1.08)
ax.pie(all_class_dist,labels=all_class_dist.index,autopct="%1.1f%%",shadow=False,startangle=90)
ax.axis('equal')
plt.savefig(root_folder + "graphs/medium_set/class_dist/all_class_dist_medium_4_classes.png")
# plt.show()
all_tracks.reset_index(inplace=True)

fig, ax = plt.subplots()
ax.set_title('Class distribution', y=1.08)
ax.pie(train_class_dist,labels=train_class_dist.index,autopct="%1.1f%%",shadow=False,startangle=90)
ax.axis('equal')
plt.savefig(root_folder + "graphs/medium_set/class_dist/train_classes_medium_4_classes.png")
# plt.show()

fig, ax = plt.subplots()
ax.set_title('Class distribution', y=1.08)
ax.pie(val_class_dist,labels=val_class_dist.index,autopct="%1.1f%%",shadow=False,startangle=90)
ax.axis('equal')
plt.savefig(root_folder + "graphs/medium_set/class_dist/val_class_dist_medium_4_classes.png")
# plt.show()

fig, ax = plt.subplots()
ax.set_title('Class distribution', y=1.08)
ax.pie(test_class_dist,labels=test_class_dist.index,autopct="%1.1f%%",shadow=False,startangle=90)
ax.axis('equal')
plt.savefig(root_folder + "graphs/medium_set/class_dist/test_class_dist_medium_4_classes.png")
# plt.show()
