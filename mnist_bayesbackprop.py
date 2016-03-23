'''
Author: Daeil Kim
Code adapted from Jan Hendrik Metzen using variational autoencoders: https://jmetzen.github.io/2015-11-27/vae.html
Tensorflow Implementation for Bayes by BackProp - http://arxiv.org/pdf/1505.05424v2.pdf
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np




'''
Class function that initializes the BayesBackprop Deep Network
'''


class BayesBackprop():
    def __init__(self, network_architecture, batch_size, learning_rate=0.01):
        self.network_architecture = network_architecture
        self.transfer_fct = tf.nn.softplus
        self.learning_rate = learning_rate

        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_features"]])
        self.y_ = tf.placeholder(tf.float32, [None, network_architecture["n_labels"]])

        weights, biases = self._initialize_variables(self.network_architecture)
        self._create_network(weights, biases)
        self._create_loss_optimizer()

        # Initialize variables and start model
        init = tf.initialize_all_variables()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def _create_network(self, weights, biases):
        # Initialize network weights and biases
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']), biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        self.y = tf.add(tf.matmul(layer_2, weights['h3_out']), biases['b3_out'])

    def _initialize_variables(self, network_architecture):
        n_features = network_architecture['n_features']
        n_hidden_units = network_architecture['n_hidden_units']
        n_labels = network_architecture['n_labels']
        weights = {
            'h1': tf.Variable(xavier_init(n_features, n_hidden_units)),
            'h2': tf.Variable(xavier_init(n_hidden_units, n_hidden_units)),
            'h3_out': tf.Variable(xavier_init(n_hidden_units, n_labels))}

        biases = {
            'b1': tf.Variable(tf.zeros([n_hidden_units], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_units], dtype=tf.float32)),
            'b3_out': tf.Variable(tf.zeros([n_labels], dtype=tf.float32))}
        return weights, biases

    def _create_loss_optimizer(self):
        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y, self.y_))
        # Use ADAM optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, batch_xs, batch_labels):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict={self.x: batch_xs, self.y_: batch_labels})
        return cost



def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


def train(network_architecture, training_epochs=10, batch_size=100, learning_rate=1e-3, display_step=1):
    # Training cycle
    bayes_backprop = BayesBackprop(network_architecture,
                             learning_rate=learning_rate,
                             batch_size=batch_size)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)

        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_labels = mnist.train.next_batch(batch_size)

            # Fit training using batch data
            cost = bayes_backprop.partial_fit(batch_xs, batch_labels)

            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            feed = {bayes_backprop.x: mnist.test.images,
                    bayes_backprop.y_: mnist.test.labels}
            print "Epoch:", '%04d' % (epoch+1), \
                  "cost = ","{:.9f}".format(avg_cost), \
                  "Accuracy: %2.3f" % (bayes_backprop.accuracy.eval(feed_dict=feed)*100)

    return bayes_backprop

if __name__ == "__main__":
    # Initialize the tensorflow graph
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    n_samples = mnist.train.num_examples

    network_architecture = \
        dict(n_hidden_units=500, # Number of hidden layer ReLu neurons
             n_features=784, # MNIST data input (img shape: 28*28)
             n_labels=10)  # number of class labels

    deep_network = train(network_architecture, training_epochs=20)


