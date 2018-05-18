import math
import numpy as np
# import h5py
# import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import csv
from tensorflow.python.framework import ops
# from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

# %matplotlib inline
np.random.seed(1)

class twitterNeuralNet():

    def __init__(self, glovePath):
        self.glove = self.loadGloveModel(glovePath)

    def loadGloveModel(self, gloveFile):
        print "Loading Glove Model"
        f = open(gloveFile,'r')
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print "Done.",len(model)," words loaded!"
        return model


    def random_mini_batches(self, X, Y, mini_batch_size = 64, seed = 0):
        """ Creates a list of random minibatches from (X, Y)
        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
        mini_batch_size -- size of the mini-batches, integer

        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y) """

        np.random.seed(seed)            # To make your "random" minibatches the same as ours
        m = X.shape[1]                  # number of training examples
        mini_batches = []

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1,m))

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k* mini_batch_size : (k+1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k* mini_batch_size : (k+1) * mini_batch_size]
    #         first_mini_batch_X = shuffled_X[:, 0 : mini_batch_size]
    #         second_mini_batch_X = shuffled_X[:, mini_batch_size : 2 * mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : num_complete_minibatches * mini_batch_size + m % mini_batch_size]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : num_complete_minibatches * mini_batch_size + m % mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches


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
        print len(scrapedData)
        scrapedData, russianData = scrapedData[1:1501], russianData[1:1501] #for now only using 1000 from each
        MAX_LEN = 50
        NUM_DIMS = 25

        XcombinedInitial = scrapedData + russianData
        numExamples = len(XcombinedInitial)
        Ycombined = [0 for i in range(len(scrapedData))] + [1 for i in range(len(russianData))]

        print 'glove-izing input'

        npX = np.zeros((numExamples, NUM_DIMS * MAX_LEN))
        for i, x in enumerate(XcombinedInitial):
            newVec = []
            for word in x:
                # print word
                if word not in self.glove:
                    newVec = newVec + [0 for j in range(NUM_DIMS)]
                else:
                    newVec = newVec + list(self.glove[word])
            newVec = newVec[:MAX_LEN*NUM_DIMS] ###TODO: FIX
            npX[i, :] = newVec

        npX = np.transpose(npX)
        print npX.shape
        print 'finished glove-izing'

        npY = np.zeros((1, numExamples))
        npY[0,:] = Ycombined
        # npY = np.transpose(npY)

        print npX
        print npY
        print 'X shape', npX.shape
        print 'Y shape', npY.shape

        #shuffle in unison

        permutation = list(np.random.permutation(numExamples))
        npX = npX[:, permutation]
        npY = npY[:, permutation]

        # numTotal = len(Xcombined)
        npX = np.split(npX, [1200, 1600], axis = 1)
        npY = np.split(npY, [1200, 1600], axis = 1)

        trainX, devX, testX = npX
        trainY, devY, testY = npY

        print 'X shape train', trainX.shape
        print 'X shape dev', devX.shape
        print 'X shape test', testX.shape

        print 'Y shape train', trainY.shape
        print 'Y shape dev', devY.shape
        print 'Y shape test', testY.shape

        return trainX, trainY, devX, devY, testX, testY

    def initialize_parameters(self):
        """ Initializes parameters to build a neural network with tensorflow. The shapes are:
                            W1 : [25, 12288]
                            b1 : [25, 1]
                            W2 : [12, 25]
                            b2 : [12, 1]
                            W3 : [6, 12]
                            b3 : [6, 1]
        Returns:
        parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3 """
        tf.set_random_seed(1)                   # so that your "random" numbers match ours
        W1 = tf.get_variable("W1", [25,1250], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
        W2 = tf.get_variable("W2", [12,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
        W3 = tf.get_variable("W3", [1,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b3 = tf.get_variable("b3", [1,1], initializer = tf.zeros_initializer())
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2,
                      "W3": W3,
                      "b3": b3}
        return parameters

    def forward_propagation(self, X, parameters):
        """ Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

        Arguments:
        X -- input dataset placeholder, of shape (input size, number of examples)
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                      the shapes are given in initialize_parameters

        Returns:
        Z3 -- the output of the last LINEAR unit """
        # Retrieve the parameters from the dictionary "parameters"
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        W3 = parameters['W3']
        b3 = parameters['b3']

        Z1 = tf.add(tf.matmul(W1, X), b1)                                              # Z1 = np.dot(W1, X) + b1
        A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)                                                # Z2 = np.dot(W2, a1) + b2
        A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        return Z3

    def compute_cost(self, Z3, Y):
        """ Computes the cost
        Arguments:
        Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
        Y -- "true" labels vector placeholder, same shape as Z3
        Returns:
        cost - Tensor of the cost function """
        # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
        print Z3
        print Y
        logits = tf.transpose(Z3)
        labels = tf.transpose(Y)
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
        print cost
        return cost

    def model(self, X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32, print_cost = True):
        """ Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
        Arguments:
        X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
        Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
        X_test -- training set, of shape (input size = 12288, number of training examples = 120)
        Y_test -- test set, of shape (output size = 6, number of test examples = 120)
        learning_rate -- learning rate of the optimization
        num_epochs -- number of epochs of the optimization loop
        minibatch_size -- size of a minibatch
        print_cost -- True to print the cost every 100 epochs
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict. """

        ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
        tf.set_random_seed(1)                             # to keep consistent results
        seed = 3                                          # to keep consistent results
        (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
        n_y = Y_train.shape[0]                            # n_y : output size
        costs = []
        print 'X_train.shape', (n_x, m)
        print 'Y_train.shape', n_y
        # Create Placeholders of shape (n_x, n_y)
        X, Y = tf.placeholder(tf.float32, name="X", shape=(n_x, None)), tf.placeholder(tf.float32, name="Y", shape=(n_y, None)) ##TODO: Fix Y Shape

        # Initialize parameters
        parameters = self.initialize_parameters()

        # Forward propagation: Build the forward propagation in the tensorflow graph
        Z3 = self.forward_propagation(X, parameters)

        # Cost function: Add cost function to tensorflow graph
        cost = self.compute_cost(Z3, Y)

        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.

        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

        # Initialize all the variables
        init = tf.global_variables_initializer()

        # Start the session to compute the tensorflow graph
        with tf.Session() as sess:
            # Run the initialization
            sess.run(init)
            # Do the training loop
            for epoch in range(num_epochs):
                epoch_cost = 0.                       # Defines a cost related to an epoch
                num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
                seed = seed + 1
                minibatches = self.random_mini_batches(X_train, Y_train, minibatch_size, seed)
                for minibatch in minibatches:
                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                    # IMPORTANT: The line that runs the graph on a minibatch.
                    # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).

                    _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                    epoch_cost += minibatch_cost / num_minibatches
                # Print the cost every epoch
                if print_cost == True and epoch % 100 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if print_cost == True and epoch % 5 == 0:
                    costs.append(epoch_cost)
            # plot the cost
            # plt.plot(np.squeeze(costs))
            # plt.ylabel('cost')
            # plt.xlabel('iterations (per tens)')
            # plt.title("Learning rate =" + str(learning_rate))
            # plt.show()
            # lets save the parameters in a variable
            parameters = sess.run(parameters)
            print ("Parameters have been trained!")
            # Calculate the correct predictions
            # print tf.argmax(Z3), tf.argmax(Y)
            # j = tf.Print(Z3, [Z3])
            # k = tf.add(j, [Y])
            # k.eval()
            correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
            print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

            return parameters



def main():
    net = twitterNeuralNet('glove/glove.twitter.27B.25d.txt')

    scrapedData, russianData = net.formatData('regTweets.csv', 'russianTweets.csv')
    trainX, trainY, devX, devY, testX, testY = net.shuffleTweets(scrapedData, russianData)
    parameters = net.model(trainX, trainY, devX, devY, learning_rate = 0.0001, num_epochs = 100, minibatch_size = 32, print_cost = True)

if __name__ == "__main__":
    main()
