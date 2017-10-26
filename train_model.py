from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from preprocess_data import preprocess_data
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras import optimizers
from config import Config
from path import Path
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset = os.listdir(Path.dir_dataset)


def training_data_generator():
    """
    Generates training data for the model.
    """
    while True:
        for index in range(len(dataset)):
            with open(Path.dir_dataset + "\\" + dataset[index], 'rb') as data:
                train_x, test_x, train_y, test_y = pickle.load(data)

                # Separates the data into batch sizes.
                batch = [x for x in range(0, (len(train_x) + 1)) if x % Config.batch_size == 0]
                count = 0

                while count < (len(batch) - 1):
                    start = batch[count]
                    end = batch[count + 1]
                    count += 1

                    training_data = (train_x[start:end], train_y[start:end])
                    yield training_data


def testing_data_generator():
    """
    Generates testing data for the model.
    """
    while True:
        for index in range(len(dataset)):
            with open(Path.dir_dataset + "\\" + dataset[index], 'rb') as data:
                train_x, test_x, train_y, test_y = pickle.load(data)

                testing_data = (test_x, test_y)
                yield testing_data


def sentiment_analysis_model():
    """
    Defines the various components in the model.
    :return: the model
    """
    with open(Path.lexicon, 'rb') as file:
        lexicon = pickle.load(file)

    model = Sequential()

    model.add(Dense(500, input_shape=(len(lexicon),), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(500, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='sigmoid'))

    adam = optimizers.Adam(lr=0.001)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def plot_graph(history):
    """
    Plots the graphs for accuracy and loss.
    :param history: output from the model
    """
    # Plot graph to display accuracy.
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper right')
    plt.savefig("model_accuracy.png")
    plt.close()

    # Plot graph to display loss.
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
    plt.savefig("model_loss.png")
    plt.close()


def train_model():
    """
    Trains the model.
    """
    # Ensure that the data has been processed already.
    preprocess_data()

    # Saves the checkpoint after every epoch.
    checkpoint_path = "checkpoints\\epoch={epoch:02d} acc={acc:.2f} loss={loss:.2f}" \
                      " val_acc={val_acc:.2f} val_loss={val_loss:.2f}.hdf5"

    checkpoints = ModelCheckpoint(checkpoint_path, verbose=True)

    # Saves the checkpoint with the smallest validation loss.
    val_checkpoint = ModelCheckpoint(Path.model, monitor='val_loss', save_best_only=True)

    # Creates the folder: checkpoints.
    if os.path.isdir(os.getcwd() + "\\checkpoints") is False:
        os.mkdir(os.getcwd() + "\\checkpoints")

    model = sentiment_analysis_model()

    steps_per_epoch = (1600000 * 0.8) / Config.batch_size

    # Train the model.
    history = model.fit_generator(generator=training_data_generator(),
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=testing_data_generator(),
                                  validation_steps=len(dataset),
                                  epochs=10,
                                  verbose=1,
                                  callbacks=[checkpoints, val_checkpoint])

    plot_graph(history)
    model.summary()


if __name__ == '__main__':

    try:
        train_model()

    except Exception:
        print("An error has occurred. Please check that your system meet the requirements or contact the author for " +
              "support.")
