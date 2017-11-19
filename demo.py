from nltk.stem import SnowballStemmer
from keras.preprocessing import text
from keras.models import load_model
from path import Path
import numpy as np
import webbrowser
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def locate_files():
    if os.path.isfile(Path.model) and os.path.isfile(Path.lexicon):
        return True

    else:
        return False


def process_input(sentence):
    # Removes special characters such as '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'.
    sentence = text.text_to_word_sequence(sentence)
    # Stem the words in the sentence
    stemmer = SnowballStemmer("english")
    sentence = [stemmer.stem(word) for word in sentence]

    with open(Path.lexicon, 'rb') as file:
        lexicon = pickle.load(file)

        # Create an numpy array that resets after every sentence.
        feature = np.array([0] * len(lexicon))

    # Determine whether the words in the sentence are found in the lexicon.
    for word in sentence:
        if word in lexicon:
            feature[lexicon.index(word)] += 1

    feature = np.reshape(feature, (1, len(lexicon)))

    return feature


def model_output(feature):
    model = load_model(Path.model)
    output = model.predict(feature)

    if output[0][0] > output[0][1]:
        return True

    else:
        return False


def demo():

    if locate_files() is False:
        print("The lexicon and/or model cannot be found. You need to train the model or download the relevant data.")
        webbrowser.open("https://github.com/LemuelHui")
        exit()

    print("This is a demonstration of a sentiment analysis neural network designed by Lemuel."
          "\nYou can enter any sentence and the model will determine whether the sentence is positive or negative."
          "\nEnter 'quit' to quit the demonstration.\n")

    while locate_files() is True:
        sentence = input()
        if sentence == 'quit':
            exit()

        else:
            feature = process_input(sentence)

            if model_output(feature) is True:
                print("The sentence is positive.\n")

            else:
                print('The sentence is negative.\n')


if __name__ == '__main__':
    try:
        demo()

    except Exception:

        print("An error has occurred. Please check that your system meet the requirements or contact the author for " +
              "support.")
