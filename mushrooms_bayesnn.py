from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm

from black_box_svi import black_box_variational_inference
from optimizers import adam

import pandas as pd
import os
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics

def load_mushroom_data():
    df = pd.read_csv(os.getcwd() + '/agaricus-lepiota.data', sep=',', header=None,
                 error_bad_lines=False, warn_bad_lines=True, low_memory=False)
    # set pandas to output all of the columns in output
    df.columns = ['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment',
             'gill-spacing','gill-size','gill-color','stalk-shape','stalk-root',
             'stalk-surf-above-ring','stalk-surf-below-ring','stalk-color-above-ring','stalk-color-below-ring',
             'veil-type','veil-color','ring-number','ring-type','spore-color','population','habitat']

    # split training from test
    X = pd.DataFrame(df, columns=df.columns[1:len(df.columns)], index=df.index)
    # put the class values (0th column) into Y
    Y = df['class']
    #y = df['class']

    le = preprocessing.LabelEncoder()
    le.fit(Y)
    y = le.transform(Y) * 5

    #have to initialize or get error below
    x_tmp = pd.DataFrame(X,columns=[X.columns[0]])

    #encode each feature column and add it to x_train (one hot encoder requires numeric input?)
    for colname in X.columns:
        le.fit(X[colname])
        #print(colname, le.classes_)
        x_tmp[colname] = le.transform(X[colname])

    oh = preprocessing.OneHotEncoder(categorical_features='all')
    oh.fit(x_tmp)
    x = oh.transform(x_tmp).toarray()
    return x, y, df



def make_nn_funs(layer_sizes, L2_reg, noise_variance, nonlinearity=np.tanh):
    """These functions implement a standard multi-layer perceptron,
    vectorized over both training examples and weight samples."""
    shapes = zip(layer_sizes[:-1], layer_sizes[1:])
    num_weights = sum((m+1)*n for m, n in shapes)

    def unpack_layers(weights):
        num_weight_sets = len(weights)
        for m, n in shapes:
            yield weights[:, :m*n]     .reshape((num_weight_sets, m, n)),\
                  weights[:, m*n:m*n+n].reshape((num_weight_sets, 1, n))
            weights = weights[:, (m+1)*n:]

    def predictions(weights, inputs):
        """weights is shape (num_weight_samples x num_weights)
           inputs  is shape (num_datapoints x D)"""
        inputs = np.expand_dims(inputs, 0)
        for W, b in unpack_layers(weights):
            outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
            inputs = nonlinearity(outputs)
        return outputs

    def logprob(weights, inputs, targets):
        log_prior = -L2_reg * np.sum(weights**2, axis=1)
        preds = predictions(weights, inputs)
        log_lik = -np.sum((preds - targets)**2, axis=1)[:, 0] / noise_variance
        return log_prior + log_lik

    return num_weights, predictions, logprob


def build_toy_dataset(n_data=100, noise_std=0.1):
    D = 1
    rs = npr.RandomState(0)
    inputs  = np.concatenate([np.linspace(-4, 4, num=n_data/2),
                              np.linspace(-8, 8, num=n_data/2)])
    targets = np.cos(inputs) + rs.randn(n_data) * noise_std
    inputs = (inputs - 4.0) / 4.0
    inputs  = inputs.reshape((len(inputs), D))
    targets = targets.reshape((len(targets), D))
    return inputs, targets



def reformat(dataset, labels):
  dataset = dataset.reshape((-1, num_features)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  # labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

if __name__ == '__main__':
    # targets are the true labels
    # inputs is the feature data set X
    #inputs, targets = build_toy_dataset()
    x, y, df = load_mushroom_data()
    train_data, test_data, train_labels, test_labels = train_test_split(x, y, test_size=0.5)
    num_features = train_data.shape[1]
    num_labels = 2
    num_units = 100
    train_data, train_labels = reformat(train_data, train_labels)
    test_data, test_labels = reformat(test_data, test_labels)

    # Specify inference problem by its unnormalized log-posterior.
    rbf = lambda x: norm.pdf(x, 0, 1)
    sq = lambda x: np.sin(x)
    num_weights, predictions, logprob = \
        make_nn_funs(layer_sizes=[num_features, 10, 10, 1], L2_reg=0.01,
                     noise_variance = 0.01, nonlinearity=rbf)

    print('# Weights: %d' % num_weights)

    batch_size = 64
    num_steps = 100
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        #log_posterior = lambda weights, t: logprob(weights, inputs, targets)
        batch_data = train_data[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        log_posterior = lambda weights, t: logprob(weights, batch_data, batch_labels)

        # Build variational objective.
        objective, gradient, unpack_params = \
            black_box_variational_inference(log_posterior, num_weights,
                                            num_samples=20)

    # Set up figure.
    '''
    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)
    '''

    def callback(params, t, g):
        print("Iteration {} lower bound {}".format(t, -objective(params, t)))

        # Sample functions from posterior.
        num_samples = 5
        rs = npr.RandomState(0)
        mean, log_std = unpack_params(params)
        #rs = npr.RandomState(0)
        sample_weights = rs.randn(num_samples, num_weights) * np.exp(log_std) + mean
        outputs = predictions(sample_weights, test_data)
        accuracy_error = np.sum(np.abs(test_labels - outputs))
        print('Accuracy Error: %.3f' % accuracy_error)
        # plot_inputs = np.linspace(-8, 8, num=400)
        # outputs = predictions(sample_weights, np.expand_dims(plot_inputs, 1))

        # Plot data and functions.
        '''
        plt.cla()
        ax.plot(inputs.ravel(), targets.ravel(), 'bx')
        ax.plot(plot_inputs, outputs[:, :, 0].T)
        ax.set_ylim([-2, 3])
        plt.draw()
        plt.pause(1.0/60.0)
        '''

    # Initialize variational parameters
    rs = npr.RandomState(0)
    init_mean    = rs.randn(num_weights)
    init_log_std = -5 * np.ones(num_weights)
    init_var_params = np.concatenate([init_mean, init_log_std])

    print("Optimizing variational parameters...")
    variational_params = adam(gradient, init_var_params,
                              step_size=0.1, num_iters=1000, callback=callback)
