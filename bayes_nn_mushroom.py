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
    y = le.transform(Y)

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


def sigmoid(x):
    return 0.5*(np.tanh(x) + 1)



def reformat(dataset, labels):
    num_labels = 2
    dataset = dataset.reshape((-1, num_features)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels


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
        eps = 1e-5
        log_prior = -L2_reg * np.sum(weights**2, axis=1)
        preds = sigmoid(predictions(weights, inputs))
        label_probabilities = targets[1,:] * np.log(preds + eps) + (1 - targets[1,:]) * np.log(1 - preds + eps)
        #log_lik = -np.sum((preds - targets)**2, axis=1)[:, 0] / noise_variance
        log_lik = -np.sum(label_probabilities, axis=(1,2))
        return log_prior + log_lik

    return num_weights, predictions, logprob






def update_nn(init_var_params, batch_data, batch_labels):
    log_posterior = lambda weights, t: logprob(weights, batch_data, batch_labels)

    # Build variational objective.
    objective, gradient, unpack_params = \
        black_box_variational_inference(log_posterior, num_weights, num_samples=20)

    variational_params = adam(gradient, init_var_params, step_size=0.01, num_iters=50)

    return variational_params





def generate_nn_output(variational_params, inputs, num_weights, num_samples):
    mean, log_std = variational_params[:num_weights], variational_params[num_weights:]
    sample_weights = rs.randn(num_samples, num_weights) * np.exp(log_std) + mean
    outputs = predictions(sample_weights, inputs)
    return outputs





if __name__ == '__main__':
    # 1 = poisonous, 0 = safe to eat
    x, y, df = load_mushroom_data()
    num_labels = 2
    num_features = x.shape[1]
    num_datums = x.shape[0]

    train_data, test_data, train_labels, test_labels = train_test_split(x, y, test_size=0.5)
    train_data, train_labels = reformat(train_data, train_labels)
    test_data, test_labels = reformat(test_data, test_labels)

    # Parameters for how we wish to learn
    batch_size = 128
    num_steps = 10000
    num_explore = 100 # number to choose random actions
    experience = []
    agent_reward = 0
    oracle_reward = 0
    cumulative_regret = 0
    cumulative_regret_over_time = []
    prev_regret = 0

    # Encode one of K actions as a one of K-vectors
    # Assume [1,0] = eat | [0,1] do not eat
    num_units = 10

    # Define the archtiecture for the nuerla network
    # Specify inference problem by its unnormalized log-posterior.
    rbf = lambda x: norm.pdf(x, 0, 1)
    relu = lambda x: np.maximum(x, 0.0)

    num_weights, predictions, logprob = \
        make_nn_funs(layer_sizes=[num_features, num_units, num_units, num_labels],
                     L2_reg=0.01,
                     noise_variance=0.01,
                     nonlinearity=relu)

    # Initialize variational parameters
    rs = npr.RandomState(0)
    num_samples = 2
    init_mean = rs.randn(num_weights)
    init_log_std = -5 * np.ones(num_weights)
    variational_params = np.concatenate([init_mean, init_log_std])



    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_data[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        variational_params = update_nn(variational_params, batch_data, batch_labels)

        if (step % 10) == 0:
            correct = 0
            num_test = len(test_labels)
            for ix, val in enumerate(test_labels):
                outputs = generate_nn_output(variational_params,
                                            np.expand_dims(test_data[ix,:],0),
                                            num_weights,
                                            num_samples)
                predicted_class = np.argmax(np.mean(outputs, axis=0))
                actual_class = np.argmax(val)
                if actual_class == predicted_class:
                    correct += 1

            print ('Accuracy at step %d: %2.3f' % (step, float(correct)/num_test*100))
