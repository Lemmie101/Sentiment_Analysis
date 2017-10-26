from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import SnowballStemmer
from keras.preprocessing import text
from config import Config
from path import Path
import urllib.request
import numpy as np
import webbrowser
import zipfile
import pickle
import random
import csv
import os


def shuffle_data():
    """
    Shuffles the raw data.
    """
    raw_data = open(Path.raw_data, "r")
    lines = raw_data.readlines()
    raw_data.close()

    random.shuffle(lines)

    raw_data = open(Path.raw_data, "w")
    raw_data.writelines(lines)
    raw_data.close()


def download_data():
    """
    Downloads, unzips, shuffles and saves the data set from Stanford.
    """
    # Download and read data set.
    print("Downloading raw data set now.")
    url = 'http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip'
    file, _ = urllib.request.urlretrieve(url)
    zip_file = zipfile.ZipFile(file, 'r')

    # Path to data set folder.
    materials = os.getcwd() + "\\materials"

    # Extract test data.
    test_data = zip_file.namelist()[0]
    zip_file.extract(test_data, path=materials)

    # Extract train data.
    train_data = zip_file.namelist()[1]
    zip_file.extract(train_data, path=materials)

    # Change file names.
    os.chdir(materials)
    os.rename("testdata.manual.2009.06.14.csv", "test_data.csv")
    os.rename("training.1600000.processed.noemoticon.csv", "train_data.csv")

    # Create a new file which combines the training data and test data because it is more convenient later.
    files = ["test_data.csv", "train_data.csv"]

    with open(Path.raw_data, 'w') as file_output:
        for file in files:
            with open(file) as file_input:
                for line in file_input:
                    file_output.write(line)

    # Shuffle data.
    shuffle_data()

    # Delete redundant files.
    for file in files:
        os.remove(materials + "\\" + file)

    print("Data set has been downloaded, unzipped, shuffled, and saved in the folder: materials.")


def extract_features_and_labels():
    """
    Extracts all the features and labels from the data set file.
    The features are saved as a list in a pickle file.
    The labels are saved as a numpy array (shape: -1, 2) in a pickle file.

    One-hot encoding is used as labels.
    "4" refers to positive statements and is encoded as [1, 0].
    "0" refers to negative statements and is encoded as [0, 1].
    """
    print("Extracting features and labels now.")

    features = []
    labels = np.array([])

    # Read the data set file and transfer the data.
    with open(Path.raw_data, 'r') as file:
        reader = csv.reader(file)

        for row in reader:

                # Adding features and labels for positive statements.
                if "4" in row[0]:
                    features.append(row[-1])
                    label = np.array([1, 0])
                    labels = np.concatenate((labels, label))

                # Adding features and labels for negative statements.
                elif "0" in row[0]:
                    features.append(row[-1])
                    label = np.array([0, 1])
                    labels = np.concatenate((labels, label))

    # Reshape the labels array.
    labels = labels.reshape(-1, 2)

    pickle.dump(features, open(Path.features, 'wb'))
    pickle.dump(labels, open(Path.labels, 'wb'))

    print("Features and labels have been extracted successfully.")


def stem_features():
    """
    Stems all the words in every sentence(example: sitting --> sit).
    """
    print("Stemming features now.")

    # The stemmed sentence will be appended to a new list: stemmed_features.
    stemmed_features = []
    # This is the stemmer used from NLTK.
    stemmer = SnowballStemmer("english")

    with open(Path.features, 'rb') as file:
        features = pickle.load(file)

    # Separate the sentence into words, stem the words and join the words back to form the sentence.
    for sentence in features:
        sentence = [stemmer.stem(word) for word in sentence.split()]
        sentence = " ".join(sentence)
        stemmed_features.append(sentence)

    pickle.dump(stemmed_features, open(Path.features, 'wb'))

    print("Features have been stemmed successfully.")


def create_lexicon():
    """
    Create a lexicon which will act as a dictionary during training.
    Words from a given sentence will be compared to the lexicon and the output will be processed by the neural network.
    """
    print("Creating lexicon now.")

    # The vectorizer will create a lexicon with up to 1000 words with common English words such as "the" eliminated.
    vectorizer = TfidfVectorizer(decode_error='replace',
                                 strip_accents='unicode',
                                 analyzer='word',
                                 stop_words='english',
                                 max_features=Config.lexicon_size,
                                 min_df=0.0005,
                                 max_df=0.3)

    with open(Path.features, 'rb') as file:
        features = pickle.load(file)

    # Activate the vectorizer and sort the lexicon alphabetically.
    vectorizer.fit_transform(features)
    lexicon = vectorizer.get_feature_names()
    lexicon = sorted(lexicon)

    print("The length of the lexicon is {} words.".format(len(lexicon)))
    print(lexicon)

    pickle.dump(lexicon, open(Path.lexicon, 'wb'))

    print("Lexicon has been created successfully.")


def split_data(count, feature_set, test_size):
    """
    Splits the data into training set and test set.
    :param count: an indication of the number of features have been processed already
    :param feature_set: the feature set
    :param test_size: the size of the test set in percentage, default = 20%
    :return train_x, test_x, train_y, test_y
    """
    print("Splitting the data between training and test set now.")

    with open(Path.labels, 'rb') as file:
        labels = pickle.load(file)
        labels = labels[(count - Config.frequency):count]

    # Split data.
    train_x, test_x, train_y, test_y = train_test_split(feature_set, labels, test_size=test_size)

    return train_x, test_x, train_y, test_y


def save_dataset(count, feature_set, lexicon):
    """
    Saves the dataset into the directory: dataset and divides the data accordingly.
    :param count: an indication of the number of features have been processed already
    :param feature_set: the feature set
    :param lexicon: the lexicon
    """
    # Reshape the feature set.
    feature_set = feature_set.reshape((Config.frequency, len(lexicon)))

    # Split data into training set and testing set.
    train_x, test_x, train_y, test_y = split_data(count, feature_set, Config.test_size)

    if os.path.isdir(Path.dir_dataset) is False:
        os.mkdir(Path.dir_dataset)

    # Create file path.
    dataset = Path.dir_dataset + "\\dataset ({} to {}).pkl".format((count - Config.frequency), count)

    # Save features.
    pickle.dump([train_x, test_x, train_y, test_y], open(dataset, 'wb'))

    print("Dataset ({} to {}) has been processed successfully.".format((count - Config.frequency), count))

    # Save progress.
    pickle.dump(count, open(Path.count, 'wb'))


def create_dataset():
    """
    Create a feature set (shape: len(features), len(lexicon)) as a numpy array.
    """
    print("Creating feature set now.")

    # Keep track of progress.
    try:
        with open(Path.count, 'rb') as file:
            count = pickle.load(file)

    except FileNotFoundError:
        count = 0

    with open(Path.features, 'rb') as file:
        features = pickle.load(file)
        # Set the total number of features as 1600000 as it is easier to divide the dataset.
        total_features = 1600000
        features = features[count: total_features]

    with open(Path.lexicon, 'rb') as file:
        lexicon = pickle.load(file)

    feature_set = np.array([])
    first_loop = True

    for sentence in features:

        # Keeps track of progress.
        if first_loop is False and count % 100 == 0:
            percentage = count / total_features * 100
            print("Sentences Processed: {}/{} = {}%".format(count, total_features, round(percentage, 2)))

        if first_loop is False and count % Config.frequency == 0:

            save_dataset(count, feature_set, lexicon)

            # Clear feature set.
            feature_set = np.array([])

        # Create an numpy array that resets after every sentence.
        x = np.array([0] * len(lexicon))
        # Removes special characters such as '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'.
        sentence = text.text_to_word_sequence(sentence)

        # Determine whether the words in the sentence are found in the lexicon.
        for word in sentence:
            if word in lexicon:
                x[lexicon.index(word)] += 1

        feature_set = np.concatenate((feature_set, x))

        count += 1

        first_loop = False

    # Save final dataset
    save_dataset(count, feature_set, lexicon)


def file_exist_in_dir(path):
    return any(os.path.isfile(os.path.join(path, i)) for i in os.listdir(path))


def combined_methods():
    """
    Executes all the methods required to pre-process all the data.
    """
    try:
        extract_features_and_labels()
        stem_features()
        create_lexicon()
        create_dataset()

    except Exception:
        print("An error has occurred. Please check that your system meet the requirements or contact the author for " +
              "support.")


def preprocess_data():

    # Check whether data set has been downloaded already
    if os.path.isfile(Path.raw_data):
        print("Found raw data set.")

    else:
        download_data()

    # Check whether features and labels have been processed already
    if os.path.isfile(Path.lexicon) and file_exist_in_dir(Path.dir_dataset) is True:
        print("Found the lexicon and data set files.")

    else:
        _ = False
        print("The raw data needs to be extracted and processed accordingly. This may take up to 8 hours." +
              "\nAlternatively, you can choose to download the processed files." +
              "\nDo you want to process the raw data? y/n")

        while _ is False:
            answer = input()

            if answer == 'y':
                _ = True
                combined_methods()

            elif answer == 'n':
                webbrowser.open("https://github.com/LemuelHui")
                exit()

            else:
                print("Please enter 'y' or 'n'.")
