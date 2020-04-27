#Music Analysis (Genre Classification) using Deep Learning
----------------------------------------------------------------------------------------------------------------------------

## Project Structure

1. music_analysis_fp <br />
    1.1 data <br />
        1.1.1 fma_metadata - consists of the metadata needed to load audio files (tids) <br />
        1.1.2 fma_small - the small dataset downloaded from the repository given in data.txt <br />
        1.1.3 npy_files - contains processed files for medium and small datasets (download from link given in data.txt) <br />
        1.1.4 weights - our trained weights are stored in crnn_20_latest.h5 <br />
    1.2 docs - contains graphs, report and some logs <br />
    1.3 fma_env - virtual env for the project <br />
    1.4 presentation - contains the project presentation <br />
    1.5 sources - various research paper and other sources used in the project <br />
    1.6 src <br />
        1.6.1 helper - contains some helper files, the main one is utils.py <br />
        1.6.2 processing - contains .py programs to help with pre-processing, cleaning metadata, conversion to npy <br />
        1.6.3 trials - contains experimental configs of the network <br />
        1.6.4 `cnn_model_using_npy_medium_set.ipynb` is the core file that loads the npy files and trains the network <br />
        1.6.5 `visualize.ipynb` is used in producing visuals for the network <br />
        
## Note
1. The config variales at the beginning of the files must be changed to use either small or medium datasets for training, plotting etc
