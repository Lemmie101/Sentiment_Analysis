import os


class Path:
    """
    Locations of the various essential files.
    """
    raw_data = os.getcwd() + '\\materials\\raw_data.csv'
    features = os.getcwd() + '\\materials\\features.pkl'
    labels = os.getcwd() + '\\materials\\labels.pkl'
    lexicon = os.getcwd() + '\\materials\\lexicon.pkl'
    dir_dataset = os.getcwd() + '\\dataset'
    count = os.getcwd() + '\\materials\\count.pkl'
    model = os.getcwd() + "\\materials\\model.hdf5"
