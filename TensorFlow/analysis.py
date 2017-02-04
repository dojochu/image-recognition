from keras.models import load_model
from tensorflow.examples.tutorials.mnist import input_data
from tools import pyp, tf,reshape, printImage, printLabel, plot_miss, plot_filters, plot_featuremap
import numpy as np

mnist = input_data.read_data_sets(train_dir = 'MNIST_data', one_hot=True)
model = load_model('modelA')

test_images = np.concatenate([reshape(x) for x in mnist.test.images])
test_labels = mnist.test.labels
pred = model.predict(x=test_images, verbose=0)
plot_miss(pred[:1000], test_labels[:1000], mnist)


plot_featuremap(model, [1], mnist.train.images[:1], np.arange(0,model.layers[1].output_shape[1]))
plot_featuremap(model, [6], mnist.train.images[[7,6,13,1,2,28,3,0,5,8]], [[0,1],[0,1]])
