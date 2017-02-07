__author__ = 'skao'

import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as pyp
import scipy.spatial as sp
import tensorflow as tf
from skimage.transform import rotate
import math

def read_data(filepath, number_of_rows=None):
    dataframe = read_csv(filepath, nrows = number_of_rows)
    dataframe['Image'] = dataframe['Image'].apply(lambda var: np.fromstring(var, sep = ' ')/255)
    dataframe = dataframe.dropna()
    X = np.vstack(dataframe['Image'])
    X = X.astype(np.float32)
    Y = dataframe[dataframe.columns[:-1]].values
    #combine = [[x,y] for x,y in zip(X,Y)]
    return X,Y


def print_image(image, ax = pyp, reshape_bool=True, flip = True):
    if reshape_bool:
        ax = ax.pcolormesh(reshape(image, flip)[0][0], cmap='bone')
    else:
        ax = ax.pcolormesh(image[0], cmap='bone')
    return ax

def create_image_display(images, pred, label, reshape_bool = True, nrows = 2, ncols = 4):
    num_frames = int(len(images)/(nrows*ncols))
    pic_ind = 0
    for frame in range(0,num_frames):
        fig, ax = pyp.subplots(nrows,ncols)
        ax = [ax] if len(ax) == 1 else ax.flatten()
        for axis in ax:
            picture = print_image(images[pic_ind], axis, reshape_bool=reshape_bool, flip=True)
            pyp.pause(0.001)
            face_color = add_face_points(picture.get_facecolors(), label[pic_ind], [0,1,0,1]) if label is not None else picture.get_facecolors()
            face_color = add_face_points(picture.get_facecolors(), pred[pic_ind], [1,0,0,1]) if pred is not None else picture.get_facecolors()
            picture.set_facecolor(face_color)
            pic_ind += 1

def add_face_points(face_map, label, color=[1,1,1,1]):
    for coord in label.reshape(15,2):
        try:
            face_map[coord_to_ind(coord[0], coord[1],96)] = color
            face_map[coord_to_ind(coord[0]+1, coord[1]+1,96)] = color
            face_map[coord_to_ind(coord[0]-1, coord[1]+1,96)] = color
            face_map[coord_to_ind(coord[0]-1, coord[1]+1,96)] = color
            face_map[coord_to_ind(coord[0]+1, coord[1]-1,96)] = color
        except IndexError:
            continue
    return face_map

def coord_to_ind(x,y, length):
    return (length-1)-int(np.floor(x))+((length-1)-int(np.floor(y)))*length

def reshape(image, flip=True):
    length = int(np.sqrt(image.shape[0]))
    return np.flipud(image).reshape(1,1,length,length) if flip else image.reshape(1,1,length,length)



def get_batch(data, labels, batch_size=None, reshape_bool=True):
    num = data.shape[0]
    if batch_size:
        sample = np.random.choice(np.arange(0,num), size=batch_size, replace=False)
        data_batch = np.vstack([ reshape(image) if reshape_bool else image for image in data[sample] ])
        label_batch = labels[sample]
    else:
        data_batch = np.vstack([ reshape(image) if reshape_bool else image for image in data])
        label_batch = labels
    return data_batch, label_batch

def distance(a, b):
    return sp.distance.euclidean(a,b)

def plot_featuremap(model, layers, images, filters = 0):
    for l in range(0,len(layers)):
        fig, axes = pyp.subplots(nrows = len(images)+1, ncols = len(filters[l])+1)
        axes = axes.flatten()
        for ax in axes:
            ax.set_xticklabels([],[])
            ax.set_yticklabels([], [])
        ax = 0
        print_image(np.ones(shape=(96,96)),ax = axes[ax],reshape_bool=False, flip=False)
        ax += 1
        for filter in filters[l]:
            if model.layers[layers[l]].name.startswith('conv'):
                print_image(model.layers[layers[l]].get_weights()[0][filter][0], ax= axes[ax], reshape_bool=False, flip=False)
                axes[ax].set_title('filter: %d' % filter)
            elif model.layers[layers[l]].name.startswith('max'):
                print_image(model.layers[layers[l]-1].get_weights()[0][filter][0], ax= axes[ax], reshape_bool=False, flip=False)
                axes[ax].set_title('filter: %d' % filter)
            else:
                print_image(np.ones(shape=(96, 96)), ax= axes[ax], reshape_bool=False, flip=False)
            ax += 1
        for image in images:
            output = reshape(image)
            for ind in range(0,layers[l]+1):
                sess = tf.Session()
                func = model.layers[ind].call(output)
                if ind == 0:
                    output = func
                    continue
                sess.run(tf.global_variables_initializer())
                with sess:
                    output = func.eval()
            print_image(image, axes[ax], reshape_bool=True, flip=True)
            ax += 1
            for filter in filters[l]:
                print_image(output[0][filter], axes[ax], False, False)
                ax += 1

def rotate_images(images, degree, reshape_bool=True):
    return np.vstack([rotate_image(images, degree, reshape_bool) for image in images])

def rotate_image(image, degree, reshape_bool = True):
    return rotate(reshape(image,False)[0][0], degree).flatten() if reshape_bool else rotate(image[0], degree).flatten()

def rotate_labels(labels, degree, center = [47,47]):
    return np.vstack([rotate_label(labels[i], degree, center) for label in labels])


def rotate_label(label, degree, center = [47,47]):
    return (apply_rotation_matrix(degree, (label.reshape(15,2) - center).transpose()).transpose() + center).flatten()

def apply_rotation_matrix(degree, coord):
    radians = -degree/180*math.pi
    rotation_mat = [[math.cos(radians), -math.sin(radians)],[math.sin(radians), math.cos(radians)]]
    return np.matmul(rotation_mat, coord)