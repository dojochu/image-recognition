from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.models import Model
from tensorflow.examples.tutorials.mnist import input_data
from tools import reshape, printImage, printLabel, plot_miss, plot_filters, plot_featuremap
import matplotlib.pyplot as pyp
import numpy as np

# Load MNIST data
mnist = input_data.read_data_sets(train_dir = 'MNIST_data', one_hot=True)

# Create placeholder nodes in TensorFlow graph for input values (images) and output values (labels)
x = Input(shape=(1, 28, 28), dtype='float32')

# Define some functions to improve initialization of weights and biases
model = Convolution2D(nb_filter=32, nb_row=5, nb_col=5, activation='relu', border_mode='same')(x)
model = MaxPooling2D(pool_size=(2,2), border_mode='valid')(model)
model = Convolution2D(nb_filter=64, nb_row=5, nb_col=5, activation='relu', border_mode='same')(model)
model = MaxPooling2D(pool_size=(2,2), border_mode='valid')(model)
#model = Convolution2D(nb_filter=5, nb_row=2, nb_col=2, activation='relu', border_mode='same')(model)
#model = MaxPooling2D(pool_size=(2,2), border_mode='valid')(model)
model = Flatten()(model)
model = Dense(output_dim=15, activation= 'softmax')(model)
model = Dropout(p=0.65)(model)
model = Dense(output_dim=10, activation= 'softmax')(model)
model = Dropout(p=0.8)(model)

# Apply new optimizer ADAM for the model
model = Model(input=x, output=model)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

batch_size = 25
batch_num = 0
test_images = np.concatenate([reshape(x) for x in mnist.test.images])
test_labels = mnist.test.labels
for i in range(1,10000):
    data = mnist.train.next_batch(batch_size)
    batch = np.concatenate([reshape(x) for x in data[0]])
    label = data[1]
    batch_num += batch_size
    if i%1000 == 0:
        print("Trained on " + str(batch_num) + " images")
        results = model.evaluate(x=test_images, y=test_labels, verbose=0, batch_size=batch_size)
        print("Cost Function: %s \n Accuracy: %s" % (results[0], results[1]))
    model.train_on_batch(batch, label)



