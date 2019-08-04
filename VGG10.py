import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2
from Layers import VanillaBatchLayer, Flatten, Conv2PoolingLayer, DropoutLayer, Conv3PoolingLayer, VanillaLayer
from datetime import datetime
import sys

class VGG10:
    def __init__(self):
        self.__ensambled = False


    def assemble(self, X_shape, Y_shape, conv_shapes=(64, 256, 512), vanilla_shapes=(4096,), activation='relu'):

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
            test_batch_number=5, epochs=5, verbose=False, best=float('inf'), show_percentage=False):

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
        assert len(conv_shapes) == 3 and len(vanilla_shapes) == 1
        self.__layers = []
        
        #First layer
        self.__layers.append(Conv2PoolingLayer("1", self.D[-1], conv_shapes[0]))

        #Second layer
        self.__layers.append(Conv3PoolingLayer("2", conv_shapes[0], conv_shapes[1]))

        #Third layer
        self.__layers.append(Conv3PoolingLayer("3", conv_shapes[1], conv_shapes[2]))

        #Fourth layer flatten
        self.__layers.append(Flatten())

        M1 = (self.D[0] * self.D[1] * conv_shapes[2]) // 64

        #Fifth layer
        self.__layers.append(VanillaBatchLayer("4", M1, vanilla_shapes[0]))

        #Last layer
        self.__layers.append(VanillaLayer("5", vanilla_shapes[0], self.K, lambda X: X))


    def save_model(self, path):
        assert self.sess_path is not None

        with tf.Session() as sess:
            self.saver.restore(sess, self.sess_path)
            self.saver.save(sess, path)


def accuracy(p, t):
    return np.mean(p == t)
