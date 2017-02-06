__author__ = 'skao'

import data_tools as dt
from keras.models import load_model
import numpy as np

model = load_model('modelC')
images, labels = dt.read_data('data/training/training.csv', 1000)
images, labels = dt.read_data('data/test/test.csv')

#==========================================
# Analyze Model
#==========================================

test_batch, label_batch = dt.get_batch(images, labels, 900)
results = model.predict(test_batch)

dist = [dt.distance(a,b) for a,b in zip(results, label_batch)]

#Best and Worst Performing
worst = np.argsort(dist)[-8:]
best = np.argsort(dist)[:8]
dt.create_image_display(test_batch[worst], results[worst], label_batch[worst], False, 2, 4)
dt.create_image_display(test_batch[best], results[best], label_batch[best], False, 2, 4)

#==========================================
# Rotations
#==========================================
rot_batch, rot_label = dt.get_batch(dt.rotate_images(images, 90, True), dt.rotate_labels(labels, 90), 900)
rot_results = model.predict(rot_batch)

rot_dist = [dt.distance(a,b) for a,b in zip(rot_results, rot_label)]

#Best and Worst Performing
rot_worst = np.argsort(rot_dist)[-8:]
rot_best = np.argsort(rot_dist)[:8]
dt.create_image_display(rot_batch[rot_worst], results[rot_best], rot_label[rot_worst], False, 2, 4)
dt.create_image_display(rot_batch[rot_best], results[rot_best], rot_label[rot_best], False, 2, 4)


dt.plot_featuremap(model, [1,3,5], images[5:10], [[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5]])
dt.plot_featuremap(model, [1], images[:20], [[6,7,8,9]])

