__author__ = 'skao'

import data_tools as dt
from keras.layers import Dense, Convolution2D, Input, MaxPooling2D, Flatten, Dropout
from keras.models import Model
import pandas as pd

#==========================================
# Reading in data
#==========================================
images, labels = dt.read_data('data/training/training.csv')
#images, labels = dt.get_batch(images, labels, None, True)
#images, labels = dt.get_batch(images, labels, 600, False)
#for angle in [45,90,135,180,-45,-90,-135]:
#    images = dt.np.vstack((images,dt.rotate_images(images, angle, reshape_bool=True)))
#    labels = dt.np.vstack((labels,dt.rotate_labels(labels, angle)))
#pd.DataFrame(images).to_csv('data/training/training_images.csv')
#pd.DataFrame(labels).to_csv('data/training/training_labels.csv')

validate_data, validate_labels = dt.read_data('data/test/test.csv')
validate_data, validate_labels = dt.get_batch(validate_data, validate_labels, validate_data.shape[0])


#==========================================
# Define Input Shape
#==========================================
input = Input(shape=(1,96,96), dtype='float32')


#==========================================
# Define Model Architecture
#==========================================
model = Convolution2D(nb_filter = 32, nb_row = 3, nb_col = 3, activation='relu', border_mode='same')(input)
model = MaxPooling2D(pool_size=(2,2), border_mode='valid')(model)
model = Convolution2D(nb_filter = 64, nb_row=5, nb_col=5, activation='relu', border_mode='same')(model)
model = MaxPooling2D(pool_size=(2,2), border_mode='valid')(model)
#model = Convolution2D(nb_filter = 128, nb_row=5, nb_col=5, activation='relu', border_mode='same')(model)
#model = MaxPooling2D(pool_size=(2,2), border_mode='valid')(model)
model = Flatten()(model)
model = Dense(output_dim=50)(model)
#model = Dropout(0.95)(model)
#model = Dense(output_dim=60)(model)
#model = Dropout(0.9)(model)
model = Dense(output_dim=30)(model)


model = Model(input=input, output=model)
model.compile(optimizer='Adadelta', loss='mean_squared_error', metrics=['mean_squared_error'])

#==========================================
# Train Model
#==========================================
batch_size = 10
epochs = 3000
angles = [0, 45, 90, 135, 180, -45, -90, -135]

for epoch in range(0,epochs):

    data_batch, label_batch = dt.get_batch(images, labels, batch_size)
    rotation = angles[dt.np.random.randintlen((angles))]
    data_batch = dt.rotate_images(data_batch, rotation, reshape_bool=False)
    label_batch = dt.rotate_labels(label_batch, rotation)
    model.train_on_batch(data_batch, label_batch)
    print('Epoch: %d' % epoch)
    if (epoch+1)%(epochs/10) == 0:
        #tb,tl = dt.get_batch(test_images, test_labels, 8)
        results = model.predict(validate_data[:8])
        dt.create_image_display(validate_data[:8], results, None, False, 2, 4)
        #accuracy = model.test_on_batch(test_images, test_labels, batch_size=batch_size)
        #print('Training Progress: Trained on %d images with Accuracy %d' % (epoch*batch_size, accuracy[0]))

model.save('modelC')

#==========================================
# Validate Model
#==========================================

a, b = dt.get_batch(images[200:208], labels[200:208])
c,d = dt.get_batch(dt.rotate_images(images[200:208], 180, True), dt.rotate_labels(labels[200:208], 180))
results = model.predict(a)
results2 = model.predict(c)
dt.create_image_display(a, results, b, False, 2, 4)
dt.create_image_display(c, results2, d, False, 2, 4)