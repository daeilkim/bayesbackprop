import pandas as pd
import os
from sklearn import preprocessing
import numpy as np

def load_mushroom_data():
    def reward_function(action_chosen, label):
        if action_chosen == 0 and label == 1:
            # we chose to eat a poisonous mushroom with probability 1/2 we get really punished
            #if np.random.rand() > 0.5:
            reward = -35
            #else:
            #reward = 0
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

    df = pd.read_csv(os.getcwd() + '/data/agaricus-lepiota.data', sep=',', header=None,
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



    return x, y, df, reward_function
