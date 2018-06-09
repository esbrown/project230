import math
import numpy as np
# import h5py
# import matplotlib.pyplot as plt
# import tensorflow as tf
import sys
import csv
# import keras
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
        scrapedData, russianData = scrapedData[1:50000], russianData[1:25000]
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

        trainX, devX, testX = np.split(data, [int(numData * .92), int(numData * .96),], axis = 0)
        trainY, devY, testY = np.split(labels, [int(numData * .92), int(numData * .96),], axis = 0)
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

        ### regular LSTM
        # model.add(LSTM(100))
        # model.add(Dense(1, activation='sigmoid'))
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        LSTMNodes = [25, 200]
        DenseNodes = [16, 32]
        DropoutNums = [.2, .4]
        batchSizes = [32, 64]

        best = 0
        params = (None, None, None, None)
        with open('hyperParams.csv', 'w') as writecsv:
            fieldnames = ['loss','test_acc', 'lstm', 'dense', 'dropout', 'batchsize']
            writer = csv.DictWriter(writecsv, fieldnames = fieldnames)
            writer.writeheader()
            for lstm in LSTMNodes:
                for dense in DenseNodes:
                    for dropout in DropoutNums:
                        for batchsize in batchSizes:
                            model = Sequential()
                            model.add(Embedding(len(word_index) + 1, self.NUM_DIMS, input_length=self.MAX_WORDS_IN_TWEET, weights=[embedding_matrix], trainable = False))
                            model.add(LSTM(lstm))
                            model.add(Dense(dense))
                            model.add(Dropout(dropout))
                            model.add(Activation('relu'))
                            model.add(Dense(1))
                            model.add(Activation('sigmoid'))
                            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                            model.fit(trainX, trainY, batch_size = batchsize, epochs = 10, shuffle = True, validation_data = (devX, devY))
                            scores_2 = model.evaluate(testX, testY, verbose=0)
                            print("Accuracy: %.2f%%" % (scores_2[1]*100))
                            print('loss: ', scores_2[0], 'lstm nodes: ', lstm, 'dense nodes: ', dense, 'dropout rate: ', dropout, 'mini batch size: ', batchsize)
                            if scores_2[1]*100 > best:
                                params = (lstm, dense, dropout, batchsize)

                            writer.writerow({'loss': scores_2[0],'test_acc': scores_2[1]*100, 'lstm': lstm, 'dense':dense, 'dropout':dropout, 'batchsize':batchsize})
        print('Best Model: ', params, 'Accuracy: ', best)


def main():
    net = twitterNeuralNet('glove/glove.twitter.27B.200d.txt')
    scrapedData, russianData = net.formatData('electionTweets.csv', 'russianTweets.csv')
    net.shuffleTweets(scrapedData, russianData)

if __name__ == "__main__":
    main()
