import math
import numpy as np
# import h5py
# import matplotlib.pyplot as plt
# import tensorflow as tf
import sys
import csv
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LSTM
# import utils
from keras.preprocessing.sequence import pad_sequences


class twitterNeuralNet():

    def __init__(self, glovePath):
        self.glove = self.loadGloveModel(glovePath)
        self.NUM_DIMS = 25
        self.MAX_WORDS_IN_TWEET = 50

    def loadGloveModel(self, gloveFile):
        print "Loading Glove Model"
        f = open(gloveFile,'r')
        model = {}
        i = 0
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = (embedding, i)
            i += 1
        print "Done.",len(model)," words loaded!"
        return model

    def formatData(self, scrapedFileName, russianFileName):
        scrapedData = []
        russianData = []
        with open(scrapedFileName, 'rU') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # print row[0]
                # print row[0].replace('[', '').replace(']', '').replace('\'', '').replace(' ', '').split(',')
                scrapedData.append(row[0].replace('[', '').replace(']', '').replace('\'', '').replace(' ', '').replace('\"', '').split(','))

        with open(russianFileName, 'rU') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                russianData.append(row[0].replace('[', '').replace(']', '').replace('\'', '').replace(' ', '').replace('\"', '').split(','))
        return scrapedData, russianData

    def shuffleTweets(self, scrapedData, russianData):
        scrapedData, russianData = scrapedData[1:1001], russianData[1:1001] #for now only using 1500 from each


    def train(self, trainX, trainY, devX, devY):
        NUM_DIMS = 25
        model = Sequential()
        model.add(LSTM(len(self.glove)+1, NUM_DIMS, input_length = 50))
        model.add(Dense(64))
        model.add(Dropout(.4))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(trainX, trainY, batch_size = 128, epochs = 5, shuffle = True)


def main():
    net = twitterNeuralNet('glove/glove.twitter.27B.25d.txt')
    scrapedData, russianData = net.formatData('regTweets.csv', 'russianTweets.csv')
    # trainX, trainY, devX, devY, testX, testY = net.shuffleTweets(scrapedData, russianData)

    model = net.train(trainX, trainY, devX, devY)

    # trainX, trainY, devX, devY, testX, testY = net.shuffleTweets(scrapedData, russianData)
    # parameters = net.model(trainX, trainY, devX, devY, learning_rate = 0.0001, num_epochs = 100, minibatch_size = 32, print_cost = True)

if __name__ == "__main__":
    main()
