import math
import numpy as np
# import h5py
# import matplotlib.pyplot as plt
# import tensorflow as tf
import sys
import csv
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Input
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
# import utils
from keras.preprocessing.sequence import pad_sequences


class twitterNeuralNet():

    def __init__(self, glovePath):
        self.glove = self.loadGloveModel(glovePath)
        self.NUM_DIMS = 25
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
        scrapedData, russianData = scrapedData[1:1001], russianData[1:1001]
        labels = [0 for i in range(len(scrapedData))] + [1 for i in range(len(russianData))]
        texts = scrapedData + russianData

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

        trainX, devX, testX = np.split(data, [1400, 1700,], axis = 0)
        trainY, devY, testY = np.split(labels, [1400, 1700,], axis = 0)
        print(trainY.shape)

        embedding_matrix = np.zeros((len(word_index) + 1, self.NUM_DIMS))

        for word, i in word_index.items():
            if word in self.glove:
            # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = self.glove[word]

        # embedding_layer = Embedding(len(word_index) + 1, self.NUM_DIMS, , input_length=self.MAX_WORDS_IN_TWEET, trainable=False)
        # sequence_input = Input(shape=(self.MAX_WORDS_IN_TWEET,), dtype='int32')
        # embedded_sequences = embedding_layer(sequence_input)

        model = Sequential()

        model.add(Embedding(len(word_index) + 1, self.NUM_DIMS, input_length=self.MAX_WORDS_IN_TWEET, weights=[embedding_matrix], trainable = False))
        model.add(LSTM(100))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(trainX, trainY, batch_size = 128, epochs = 5, shuffle = True)
        scores = model.evaluate(devX, devY, verbose=0)
        scores_2 = model.evaluate(testX, testY, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
        print("Accuracy: %.2f%%" % (scores_2[1]*100))


    # def train(self, trainX, trainY, devX, devY):
    #     model = Sequential()
    #     model.add(LSTM(100))
    #     model.add(Dense(64))
    #     model.add(Dropout(.4))
    #     model.add(Activation('relu'))
    #     model.add(Dense(1))
    #     model.add(Activation('sigmoid'))
    #     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #     model.fit(trainX, trainY, batch_size = 128, epochs = 5, shuffle = True)


def main():
    net = twitterNeuralNet('glove/glove.twitter.27B.25d.txt')
    scrapedData, russianData = net.formatData('regTweets.csv', 'russianTweets.csv')
    net.shuffleTweets(scrapedData, russianData)
    # model = net.train(trainX, trainY, devX, devY)

if __name__ == "__main__":
    main()
