#Music Analysis (Genre Classification) using Deep Learning
----------------------------------------------------------------------------------------------------------------------------

## Project Structure

``|- music_analysis_fp
    |
    |- data
        |- fma_metadata - consists of the metadata needed to load audio files (tids)
        |- fma_small - the small dataset downloaded from the repository given in data.txt
        |- npy_files - contains processed files for medium and small datasets (download from link given in data.txt)
        |- weights - our trained weights are stored in crnn_20_latest.h5
    |- docs - contains graphs, report and some logs
    |- fma_env - virtual env for the project
    |- presentation - contains the project presentation
    |- sources - various research paper and other sources used in the project
    |- src
        |- helper - contains some helper files, the main one is utils.py 
        |- processing - contains .py programs to help with pre-processing, cleaning metadata, conversion to npy
        |- trials - contains experimental configs of the network
        |- `cnn_model_using_npy_medium_set.ipynb` is the core file that loads the npy files and trains the network
        |- `visualize.ipynb` is used in producing visuals for the network``
        
## Note
1. The config variales at the beginning of the files must be changed to use either small or medium datasets for training, plotting etc
