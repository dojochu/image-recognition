__author__ = 'skao'

import numpy as np
import os
from pandas.io.parsers import read_csv




training_set = '~/PycharmProjects/DeepLearning/Data/training/training.csv'
testing_set = '~/PycharmProjects/DeepLearning/Data/testing/testing.csv'

#def load(test=False, cols = None):

filename = training_set
dataframe = read_csv(os.path.expanduser(filename))

dataframe['Image'] = dataframe['Image'].apply(lambda var: np.fromstring(var, sep = ' '))

dataframe = dataframe.dropna()

X = np.vstack(dataframe['Image'].values)/255
X = X.astype(np.float32)

Y = (dataframe[dataframe.columns[:-1]].values - 48)/48

from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

net1 = NeuralNet(
        layers = [
            ('input', layers.Inputlayer),
            ('hidden', layers.DenseLayer),
            ('output', layers.DenseLayer)
            ],
        input_shape=(None, 9216),
    hidden_num_units = 100,
    output_nonlinearity=None,
    output_num_units = 30,
    update = nesterov_momentum,
    update_learning_rate = 0.01,
    update_momentum = 0.9,
    regression=True,
    max_epochs = 400,
    verbose=1,)

net1.fit(X,Y)

)