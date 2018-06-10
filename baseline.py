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
# import matplotlib.pyplot as plt


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
        print(len(scrapedData), len(russianData))
        scrapedData, russianData = scrapedData[1:20000], russianData[1:20000]
        labels = [0 for i in range(len(scrapedData))] + [1 for i in range(len(russianData))]
        texts = scrapedData + russianData

        numData = len(labels)

        x = []
        for tweet in texts:
            gloveTweet = np.zeros(self.MAX_WORDS_IN_TWEET*self.NUM_DIMS)
            for i, word in enumerate(tweet):
                if word in self.glove and i < 60:
                    gloveTweet[i*self.NUM_DIMS:(i+1)*self.NUM_DIMS] = self.glove[word]
            x.append(gloveTweet)

        data = np.array(x)
        labels = np.array(labels)

        print('Shape of data tensor:', data.shape)
        print('Shape of label tensor:', labels.shape)

        permutation = list(np.random.permutation(data.shape[0]))
        data = data[permutation]
        labels = labels[permutation]
        print(labels.shape)

        testX, devX, trainX = np.split(data, [int(numData * .02), int(numData * .04),], axis = 0)
        testY, devY, trainY = np.split(labels, [int(numData * .02), int(numData * .04),], axis = 0)
        print(trainY.shape)

        model = Sequential()
        model.add(Dense(10, activation = 'relu', input_shape=(self.MAX_WORDS_IN_TWEET*self.NUM_DIMS,)))
        model.add(Dropout(0.5, noise_shape = None, seed = None))
        model.add(Dense(5, activation = 'relu'))
        model.add(Dropout(0.5, noise_shape = None, seed = None))
        model.add(Dense(1, activation = 'sigmoid'))
        model.summary()
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        results = model.fit(trainX,trainY, epochs = 5, batch_size = 128, validation_data = (devX, devY))
        scores_2 = model.evaluate(testX, testY, verbose=0)
        print("Accuracy: %.2f%%" % (scores_2[1]*100))



        


def main():
    net = twitterNeuralNet('glove/glove.twitter.27B.25d.txt')
    scrapedData, russianData = net.formatData('electionTweets.csv', 'russianTweets.csv')
    net.shuffleTweets(scrapedData, russianData)

if __name__ == "__main__":
    main()
