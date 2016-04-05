from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import numpy as np
from keras.datasets import mnist
from keras.layers.core import Lambda
from keras import backend as K
import input_data
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split


# Parameters for how we wish to learn
batch_size = 128
num_steps = 100000
num_explore = 100 # number to choose random actions
experience = []
agent_reward = 0
oracle_reward = 0
cumulative_regret = 0
cumulative_regret_over_time = []
cumulative_index = []
prev_regret = 0

# Encode one of K actions as a one of K-vectors
# Assume [1,0] = eat | [0,1] do not eat
num_units = 30
num_actions = 2
x, y, df, reward_function = input_data.load_mushroom_data()
num_features = x.shape[1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
num_datums = x_train.shape[0]
F = num_features + num_actions

model = Sequential()
model.add(Dense(100, input_shape=(F,)))
model.add(Activation('relu'))
model.add(Lambda(lambda x: K.dropout(x, level=0.2)))
model.add(Dense(100, init='glorot_uniform'))
model.add(Activation('relu'))
model.add(Lambda(lambda x: K.dropout(x, level=0.2)))
model.add(Dense(1))

rms = RMSprop()
model.compile(loss='mean_squared_error', optimizer=rms)

action_matrix = np.eye(num_actions)

# Explore and take random actions
num_random_steps = 100
num_actions = 2
action_matrix = np.eye(num_actions)
experience_context = []
experience_rewards = []

# Set up figure.
fig = plt.figure(figsize=(8,8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show(block=False)

print('Exploring randomly for %s number of steps' % num_random_steps)
for rr in xrange(num_random_steps):
    #Randomly select an action and receive its specified reward
    action_index = np.random.randint(0,1,1)[0]
    action = action_matrix[action_index,:]

    #Randomly select a datapoint
    datum_id = np.random.randint(0, num_datums)
    ytrue = y_train[datum_id]
    reward, oracle_reward = reward_function(action_index, ytrue)

    # Store the experience of that reward as a training/data pair
    context = np.hstack((x_train[datum_id, :], action))
    experience_context.append(context)
    experience_rewards.append(reward)

    # Calculate the cumulative regret
    cumulative_regret += oracle_reward - agent_reward



taking_a_chance = 0
playing_it_safe = 0
for step in range(num_steps):
    # train model
    model.fit(np.array(experience_context),
              np.array(experience_rewards),
              batch_size=batch_size,
              nb_epoch=1,
              verbose=1)

    #Randomly select a datapoint
    datum_id = np.random.randint(0, num_datums)
    rewards = []
    for aa in range(num_actions):
        context = np.hstack((x_train[datum_id, :], action_matrix[aa,:]))
        predictions = model.predict(context[np.newaxis,:])[0][0]
        rewards.append(predictions)

    action_chosen = np.argmax(rewards)
    if action_chosen == 0:
        taking_a_chance += 1
    else:
        playing_it_safe += 1
    reward, oracle_reward = reward_function(action_chosen, y[datum_id])

    experience_context.append(context)
    experience_rewards.append(reward)

    # Calculate the cumulative regret
    cumulative_regret += oracle_reward - agent_reward

    if (step%100 == 0):
        cumulative_regret_over_time.append(cumulative_regret)
        cumulative_index.append(step)
        print("Cumulative Regret at Step %d: %d" % (step, cumulative_regret))
        print("Number of Times we took a chance: %d" % (taking_a_chance))
        print("Number of Times we didn't: %d" % (playing_it_safe))
        print("Regret Delta at Step %d: %d" % (step, cumulative_regret-prev_regret))
        prev_regret = cumulative_regret
        plt.cla()
        ax.plot(cumulative_regret_over_time)
        plt.draw()
