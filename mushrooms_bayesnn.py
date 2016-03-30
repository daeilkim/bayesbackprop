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

def update_nn(init_var_params, batch_data, batch_labels):
    log_posterior = lambda weights, t: logprob(weights, batch_data, batch_labels)

    # Build variational objective.
    objective, gradient, unpack_params = \
        black_box_variational_inference(log_posterior, num_weights, num_samples=20)

    variational_params = adam(gradient, init_var_params, step_size=0.01, num_iters=1)

    return variational_params


def generate_nn_output(variational_params, inputs, num_weights, num_samples):
    mean, log_std = variational_params[:num_weights], variational_params[num_weights:]
    sample_weights = rs.randn(num_samples, num_weights) * np.exp(log_std) + mean
    outputs = predictions(sample_weights, inputs)
    return outputs






def reward_function(action_chosen, label):
    if action_chosen == 0 and label == 1:
        # we chose to eat a poisonous mushroom with probability 1/2 we get really punished
        if npr.rand() > 0.5:
            reward = -35
        else:
            reward = 5
    elif action_chosen == 0 and label == 0:
        reward = 5
    else:
        # we chose not to eat, so we get no reward
        reward = 0

    if label == 1:
        oracle_reward = 0
    else:
        oracle_reward = 5
    return reward, oracle_reward





if __name__ == '__main__':
    # 1 = poisonous, 0 = safe to eat
    x, y, df = load_mushroom_data()
    num_features = x.shape[1]
    num_datums = x.shape[0]

    # Parameters for how we wish to learn
    batch_size = 128
    num_steps = 10000
    num_explore = 100 # number to choose random actions
    experience = []
    agent_reward = 0
    oracle_reward = 0
    cumulative_regret = 0
    cumulative_regret_over_time = []

    # Encode one of K actions as a one of K-vectors
    # Assume [1,0] = eat | [0,1] do not eat
    num_units = 50
    num_actions = 2
    action_vectors = np.identity(num_actions)

    # Define the archtiecture for the nuerla network
    # Specify inference problem by its unnormalized log-posterior.
    rbf = lambda x: norm.pdf(x, 0, 1)

    num_weights, predictions, logprob = \
        make_nn_funs(layer_sizes=[num_features+num_actions, num_units, num_units, 1],
                     L2_reg=0.01,
                     noise_variance=0.01,
                     nonlinearity=rbf)

    # Initialize variational parameters
    rs = npr.RandomState(0)
    num_samples = 5
    init_mean = rs.randn(num_weights)
    init_log_std = -5 * np.ones(num_weights)
    variational_params = np.concatenate([init_mean, init_log_std])

    # Set up figure.
    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)

    for step in range(num_steps):
        # Grab a random datum
        datum_id = npr.randint(0, num_datums)

        # Assess expected reward across all possible actions (loop over context + action vectors)
        rewards = []
        contexts = np.zeros((num_actions, num_features+num_actions))
        for aa in range(num_actions):
            contexts[aa,:] = np.hstack((x[datum_id, :], action_vectors[aa,:]))
            outputs = generate_nn_output(variational_params,
                                         np.expand_dims(contexts[aa,:],0),
                                         num_weights,
                                         num_samples)
            rewards.append(np.mean(outputs))

        # Check which is greater and choose that [1,0] = eat | [0,1] do not eat
        # If argmax returns 0, then we eat, otherwise we don't
        action_chosen = np.argmax(rewards)
        reward, oracle_reward = reward_function(action_chosen, y[datum_id])

        # Calculate the cumulative regret
        cumulative_regret += oracle_reward - agent_reward
        cumulative_regret_over_time.append(cumulative_regret)

        if (step % 100) == 0:
            print("Cumulative Regret at Step %d: %d" % (step, cumulative_regret))

            plt.cla()
            ax.plot(cumulative_regret_over_time)
            plt.draw()
            plt.pause(1.0/60.0)

        # Store the experience of that reward as a training/data pair
        experience.append([contexts[action_chosen, :], reward])

        # Choose the action that maximizes the expected reward or go with epsilon greedy
        if len(experience) > batch_size:
            batch_data = np.zeros((batch_size, num_features+num_actions))
            batch_labels = np.zeros((batch_size, 1))
            indices = np.random.choice(len(experience), batch_size, replace=False)
            for ix in range(batch_size):
                ind = indices[ix]
                batch_data[ix, :] = experience[ix][0]
                batch_labels[ix, :] = experience[ix][1]
            variational_params = update_nn(variational_params, batch_data, batch_labels)



