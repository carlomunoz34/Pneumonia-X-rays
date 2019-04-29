import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2
from Layers import VanillaBatchLayer, Flatten, Conv2Batch2PoolingLayer, DropoutLayer, VanillaLayer, Conv2BatchPoolingLayer
from datetime import datetime
import sys

class ConvolutionalNetwork:
    def __init__(self):
        self.__ensambled = False


    def ensamble(self, X_shape, Y_shape, conv_shapes=(64, 128), vanilla_shapes=(4096, 4096), activation='relu'):

        self.sess_path = ""
        self.D = X_shape
        self.K = Y_shape

        if activation == 'relu':
            self.activation = tf.nn.relu

        elif activation == 'tanh':
            self.activation = tf.nn.tanh

        elif activation == 'sigmoid':
            self.activation = tf.nn.sigmoid

        #Ensamble all the layers of the NN
        self.__ensamble_net(conv_shapes, vanilla_shapes)

        #Create placeholders 
        self.X = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.D))
        self.Y = tf.placeholder(dtype=tf.int32, shape=(None,))

        self.P = self.__forward(self.X)

        #Tensorflow functions
        self.cost_op = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.P,
                labels=self.Y
            )
        )

        self.__predict_op = tf.argmax(self.P, axis=1)

        self.__predict_p = self.__forward(self.X, training=False)
        self.__predict = tf.argmax(self.__predict_p, axis=1)

        #Saver
        self.saver = tf.train.Saver()

        self.__ensambled = True    


    def fit(self, next_batch, next_test_batch, learning_rate=0.001, beta1=0.9, beta2=0.999, batch_number=100, 
            test_batch_number=5, epochs=5, verbose=True, best=float('inf'), show_percentage=True):

        assert self.__ensambled
        #Train
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2).minimize(self.cost_op)

        init = tf.global_variables_initializer()
        costs_test = []
        accuracies_test = []
        costs_train = []
        accuracies_train = []

        if verbose:
            print("Starting training")

        final_test_cost = 0

        t0 = datetime.now()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(epochs):

                epoch_train_cost = 0
                epoch_train_acc = 0
                average_time = datetime.now() - datetime.now()
                msg_len = 0

                t0_epoch = datetime.now()
                for batch in range(batch_number):
                    t0_batch = datetime.now()
                    Xtrain, Ytrain = next_batch(batch)
                    sess.run(self.train_op, feed_dict={self.X: Xtrain, self.Y:Ytrain})
        

                    if verbose:
                        train_p = sess.run(self.__predict_op, feed_dict={self.X: Xtrain})
                        train_cost = sess.run(self.cost_op, feed_dict={self.X: Xtrain, self.Y: Ytrain})
                        train_acc = accuracy(train_p, Ytrain)

                        epoch_train_cost += train_cost
                        epoch_train_acc += train_acc / batch_number

                        costs_train.append(train_cost)
                        accuracies_train.append(train_acc)

                        del Xtrain, Ytrain
                        t1_batch = datetime.now()

                        percentage = 100 * (batch + 1) / batch_number
                        time = t1_batch - t0_batch
                        message = "Batch %i of %i, %.2f" %(batch+1, batch_number, percentage) + "% " + "Time: " + str(time) + " Accuracy: %.4f, Loss: %.4f" %(train_acc, train_cost)

                        average_time += time / batch_number

                        if 0 < batch:
                            sys.stdout.write("\r" + " " * msg_len)
                            sys.stdout.write("\r" + message)

                        else:
                            sys.stdout.write(message)
                        
                        msg_len = len(message)
                        sys.stdout.flush()

                    elif show_percentage:
                        percentage = 100 * (epoch * batch_number + batch+1) / (epochs * batch_number)
                        if batch == 0 and epoch == 0:
                            print(str(percentage) + "%")
                        else:
                            sys.stdout.write("\r    ")
                            sys.stdout.write("\r" + str(percentage) + "%")

                #Test
                if verbose:
                    test_cost = 0
                    test_acc = 0
                    for i in range(test_batch_number):
                        Xtest, Ytest = next_test_batch(i)
                        
                        test_p = sess.run(self.__predict, feed_dict={self.X: Xtest})

                        test_cost += sess.run(self.cost_op, feed_dict={self.X: Xtest, self.Y: Ytest})
                        test_acc += accuracy(test_p, Ytest) / test_batch_number
                    
                    costs_test.append(test_cost)
                    accuracies_test.append(test_acc)

                    t1_epoch = datetime.now()

                    print()
                    print("Epoch %i: Train Loss: %.4f, Train Accuracy = %.4f, Test Loss = %.4f, Test Accuracy = %.4f"  %(epoch+1, epoch_train_cost, epoch_train_acc, test_cost, test_acc), 
                            "Total time = ", (t1_epoch - t0_epoch), "Average time per bach :", average_time)
            
            #Final test cost
            for i in range(test_batch_number):
                Xtest, Ytest = next_test_batch(i)
                final_test_cost += sess.run(self.cost_op, feed_dict={self.X: Xtest, self.Y: Ytest})
            
            if final_test_cost < best:
                #Save the model
                self.sess_path = "../Models/best/best_model.ckpt"
                self.saver.save(sess, self.sess_path)
        
        t1 = datetime.now()

        if verbose:
            print("Train finished")
            print("Elapsed time", t1 - t0)

        if verbose:
            return costs_train, accuracies_train, costs_test, accuracies_test
        
        return final_test_cost


    def restore_session(self, path):
        self.sess_path = path

    def predict(self, X):
        assert self.sess_path is not None

        with tf.Session() as sess:
            self.saver.restore(sess, self.sess_path)
            prediction = sess.run(self.__predict, feed_dict={self.X: X})
        
        return prediction


    def __forward(self, X, training=True):
        Z = X
        for layer in self.__layers:
            Z = layer.forward(Z, training)

        return Z


    def __ensamble_net(self, conv_shapes, vanilla_shapes):
        self.__layers = []
        
        #Add convolutional layers
        M1 = self.D[-1]
        id = 1
        for M2 in conv_shapes:
            self.__layers.append(Conv2BatchPoolingLayer(str(id), M1, M2, self.activation))
            self.__layers.append(DropoutLayer(0.25))
            id += 1
            M1 = M2

        #Flatten layer
        self.__layers.append(Flatten())

        #Feedforward layers
        M1 = (self.D[0] * self.D[1]) // (2 ** (2 * len(conv_shapes))) * M2

        id = 6
        for M2 in vanilla_shapes:
            self.__layers.append(VanillaBatchLayer(str(id), M1, M2))
            self.__layers.append(DropoutLayer(0.25))
            id += 1
            M1 = M2

        #Last layer
        self.__layers.append(VanillaLayer(str(id), M1, self.K, lambda X: X))


    def save_model(self, path):
        assert self.sess_path is not None

        with tf.Session() as sess:
            self.saver.restore(sess, self.sess_path)
            self.saver.save(sess, path)


def accuracy(p, t):
    return np.mean(p == t)



class ConvolutionalNetwork2:
    def __init__(self):
        self.__ensambled = False


    def ensamble(self, X_shape, conv_shapes=(64, 128), vanilla_shapes=(4096, 4096), activation='relu'):
        tf.reset_default_graph()
        self.sess_path = ""
        self.D = X_shape
        self.K = 1

        if activation == 'relu':
            self.activation = tf.nn.relu

        elif activation == 'tanh':
            self.activation = tf.nn.tanh

        elif activation == 'sigmoid':
            self.activation = tf.nn.sigmoid

        #Ensamble all the layers of the NN
        self.__ensamble_net(conv_shapes, vanilla_shapes)

        #Create placeholders 
        self.X = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.D))
        self.Y = tf.placeholder(dtype=tf.float32, shape=(None, 1))

        self.P = self.__forward(self.X)

        self.cost_op = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.P,
                labels=self.Y
            )
        )

        self.__predict_op = self.P

        self.__predict_p = tf.round(self.__forward(self.X, training=False))
        self.__predict = tf.round(self.__predict_p)

        #Saver
        self.saver = tf.train.Saver()

        self.__ensambled = True    


    def fit(self, next_batch, next_test_batch, learning_rate=0.001, beta1=0.9, beta2=0.999, batch_number=100, 
            test_batch_number=5, epochs=5, verbose=True, best=float('inf'), show_percentage=True):

        assert self.__ensambled
        #Train
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2).minimize(self.cost_op)

        init = tf.global_variables_initializer()
        costs_test = []
        accuracies_test = []
        costs_train = []
        accuracies_train = []

        if verbose:
            print("Starting training")

        final_test_cost = 0

        t0 = datetime.now()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(epochs):

                epoch_train_cost = 0
                epoch_train_acc = 0
                average_time = datetime.now() - datetime.now()
                msg_len = 0

                t0_epoch = datetime.now()
                for batch in range(batch_number):
                    t0_batch = datetime.now()
                    Xtrain, Ytrain = next_batch(batch)
                    sess.run(self.train_op, feed_dict={self.X: Xtrain, self.Y:Ytrain})


                    if verbose:
                        train_p = sess.run(self.__predict_op, feed_dict={self.X: Xtrain})
                        train_cost = sess.run(self.cost_op, feed_dict={self.X: Xtrain, self.Y: Ytrain})
                        train_acc = accuracy(train_p, Ytrain)

                        epoch_train_cost += train_cost
                        epoch_train_acc += train_acc / batch_number

                        costs_train.append(train_cost)
                        accuracies_train.append(train_acc)

                        del Xtrain, Ytrain
                        t1_batch = datetime.now()

                        percentage = 100 * (batch + 1) / batch_number
                        time = t1_batch - t0_batch
                        message = "Batch %i of %i, %.2f" %(batch+1, batch_number, percentage) + "% " + "Time: " + str(time) + " Accuracy: %.4f, Loss: %.4f" %(train_acc, train_cost)

                        average_time += time / batch_number

                        if 0 < batch:
                            sys.stdout.write("\r" + " " * msg_len)
                            sys.stdout.write("\r" + message)

                        else:
                            sys.stdout.write(message)
                        
                        msg_len = len(message)
                        sys.stdout.flush()
                    
                    elif show_percentage:
                        percentage = 100 * (epoch * batch_number + batch+1) / (epochs * batch_number)
                        sys.stdout.write("\r%.2f"%(percentage) + "%")

                #Test
                if verbose:
                    test_cost = 0
                    test_acc = 0
                    for i in range(test_batch_number):
                        Xtest, Ytest = next_test_batch(i)
                        
                        test_p = sess.run(self.__predict, feed_dict={self.X: Xtest})

                        test_cost += sess.run(self.cost_op, feed_dict={self.X: Xtest, self.Y: Ytest})
                        test_acc += accuracy(test_p, Ytest) / test_batch_number
                    
                    costs_test.append(test_cost)
                    accuracies_test.append(test_acc)

                    t1_epoch = datetime.now()

                    print()
                    print("Epoch %i: Train Loss: %.4f, Train Accuracy = %.4f, Test Loss = %.4f, Test Accuracy = %.4f"  %(epoch+1, epoch_train_cost, epoch_train_acc, test_cost, test_acc), 
                            "Total time = ", (t1_epoch - t0_epoch), "Average time per bach :", average_time)
            
            print("")
            #Final test cost
            for i in range(test_batch_number):
                Xtest, Ytest = next_test_batch(i)
                final_test_cost += sess.run(self.cost_op, feed_dict={self.X: Xtest, self.Y: Ytest})
            
            if final_test_cost < best:
                #Save the model
                self.sess_path = "../Models/best/best_model.ckpt"
                self.saver.save(sess, self.sess_path)
        
        t1 = datetime.now()

        if verbose:
            print("Train finished")
            print("Elapsed time", t1 - t0)

        if verbose:
            return costs_train, accuracies_train, costs_test, accuracies_test
        
        return final_test_cost


    def restore_session(self, path):
        self.sess_path = path

    def predict(self, X):
        assert self.sess_path is not None

        with tf.Session() as sess:
            self.saver.restore(sess, self.sess_path)
            prediction = sess.run(self.__predict, feed_dict={self.X: X})
        
        return prediction


    def __forward(self, X, training=True):
        Z = X
        for layer in self.__layers:
            Z = layer.forward(Z, training)

        return Z


    def __ensamble_net(self, conv_shapes, vanilla_shapes):
        self.__layers = []
        
        #Add convolutional layers
        M1 = self.D[-1]
        id = 1
        for M2 in conv_shapes:
            self.__layers.append(Conv2BatchPoolingLayer(str(id), M1, M2, self.activation))
            self.__layers.append(DropoutLayer(0.25))
            id += 1
            M1 = M2

        #Flatten layer
        self.__layers.append(Flatten())

        #Feedforward layers
        M1 = (self.D[0] * self.D[1]) // (2 ** (2 * len(conv_shapes))) * M2

        id = 6
        for M2 in vanilla_shapes:
            self.__layers.append(VanillaBatchLayer(str(id), M1, M2))
            self.__layers.append(DropoutLayer(0.40))
            id += 1
            M1 = M2

        #Last layer
        #self.__layers.append(VanillaLayer(str(id), M1, self.K, lambda X: X))
        self.__layers.append(VanillaLayer(str(id), M1, self.K, tf.nn.sigmoid))


    def save_model(self, path):
        assert self.sess_path is not None

        with tf.Session() as sess:
            self.saver.restore(sess, self.sess_path)
            self.saver.save(sess, path)
