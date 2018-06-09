import math
import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Input
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class twitterNeuralNet():

    def __init__(self, glovePath):
        self.glove = self.loadGloveModel(glovePath)
        self.NUM_DIMS = 200
        self.MAX_WORDS_IN_TWEET = 60

    def loadGloveModel(self, gloveFile):
        print ("Loading Glove Model")
        f = open(gloveFile,'r')
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print ("Done.",len(model)," words loaded!")
        return model

    def formatData(self, scrapedFileName, russianFileName):
        scrapedData = []
        russianData = []
        with open(scrapedFileName, 'rU') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                scrapedData.append(" ".join(row[0].replace('[', '').replace(']', '').replace('\'', '').replace(' ', '').replace('\"', '').lower().split(',')))

        with open(russianFileName, 'rU') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                russianData.append(" ".join(row[0].replace('[', '').replace(']', '').replace('\'', '').replace(' ', '').replace('\"', '').lower().split(',')))
        return scrapedData, russianData

    def shuffleTweets(self, scrapedData, russianData):
        print(len(scrapedData), len(russianData))
        scrapedData, russianData = scrapedData[1:1000001], russianData[1:200001]
        labels = [0 for i in range(len(scrapedData))] + [1 for i in range(len(russianData))]
        texts = scrapedData + russianData

        numData = len(labels)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        word_index = tokenizer.word_index

        data = pad_sequences(sequences, maxlen=self.MAX_WORDS_IN_TWEET)
        labels = np.asarray(labels)
        print('Shape of data tensor:', data.shape)
        print('Shape of label tensor:', labels.shape)

        permutation = list(np.random.permutation(data.shape[0]))
        data = data[permutation, :]
        labels = labels[permutation]
        print(labels.shape)

        trainX, devX, testX = np.split(data, [int(numData * .98), int(numData * .99),], axis = 0)
        trainY, devY, testY = np.split(labels, [int(numData * .98), int(numData * .99),], axis = 0)
        print(trainY.shape)

        embedding_matrix = np.zeros((len(word_index) + 1, self.NUM_DIMS))

        for word, i in word_index.items():
            if word in self.glove:
            # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = self.glove[word]

        model = Sequential()
        model.add(Embedding(len(word_index) + 1, self.NUM_DIMS, input_length=self.MAX_WORDS_IN_TWEET, weights=[embedding_matrix], trainable = False))
        model.add(LSTM(130))
        model.add(Dense(6))
        model.add(Dropout(.25))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(trainX, trainY, batch_size = 128, epochs = 10, shuffle = True, validation_data = (devX, devY))
        scores_2 = model.evaluate(testX, testY, verbose=0)
        print("Accuracy: %.2f%%" % (scores_2[1]*100))
        model.save('twitterModel.h5')

        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


def main():
    net = twitterNeuralNet('glove/glove.twitter.27B.200d.txt')
    scrapedData, russianData = net.formatData('electionTweets.csv', 'russianTweets.csv')
    net.shuffleTweets(scrapedData, russianData)

if __name__ == "__main__":
    main()
