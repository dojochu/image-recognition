
import matplotlib.pyplot as pyp
import tensorflow as tf
import numpy as np

def printImage(mnist, image_ind=0, ax = pyp):
    ax.pcolormesh(np.flipud(reshape(mnist.images[image_ind])[0][0]), cmap='gray')


def printLabel(mnist, image_ind=0):
    print(mnist.labels[image_ind])

def reshape(image):
    return image.reshape(1,1,28,28)

def plot_miss(pred, labels, mnist):

    #pred = [np.where(p == max(p), 1, 0) for p in pred]
    labels = [np.argmax(l) for l in labels]
    pred = [np.argmax(p) for p in pred]
    miss = [0 if p == l else 1 for p,l in zip(pred, labels)]
    num_plots = sum(miss)
    ind = np.nonzero(miss)[0]
    fig1, axes = pyp.subplots(nrows=int(num_plots/4)+1, ncols=4)
    axes = axes.flatten()
    for i in range(0,len(ind)):
        printImage(mnist.test, ind[i], axes[i])
        axes[i].set_title('Model: %s | Truth: %s' % (str(pred[ind[i]]), str(labels[ind[i]])))
        axes[i].set_xticks([],[])
        axes[i].set_yticks([], [])

def plot_filters(conv_layer, filter = 0, ax = pyp):
    ax.pcolormesh(conv_layer.get_weights()[0][filter][0], cmap='gray')
    ax.set_title('filter: %d' % filter)

def plot_featuremap(model, layers, images, filters = 0):
    for l in range(0,len(layers)):
        fig, axes = pyp.subplots(nrows = len(images)+1, ncols = len(filters[l])+1)
        axes = axes.flatten()
        for ax in axes:
            ax.set_xticklabels([],[])
            ax.set_yticklabels([], [])
        ax = 0
        axes[ax].pcolormesh(np.flipud(np.ones(shape=(28, 28))), cmap='gray')
        ax += 1
        for filter in filters[l]:
            if model.layers[layers[l]].name.startswith('conv'):
                plot_filters(model.layers[layers[l]], filter, axes[ax])
            elif model.layers[layers[l]].name.startswith('max'):
                plot_filters(model.layers[layers[l]-1], filter, axes[ax])
            else:
                axes[ax].pcolormesh(np.flipud(np.ones(shape=(28, 28))), cmap='gray')
            ax += 1
        for image in images:
            output = reshape(image)
            for ind in range(0,layers[l]+1):
                sess = tf.Session()
                print('ind: %d + l: %d ' % (ind, l))
                func = model.layers[ind].call(output)
                if ind == 0:
                    output = func
                    continue
                sess.run(tf.global_variables_initializer())
                with sess:
                    output = func.eval()
            axes[ax].pcolormesh(np.flipud(reshape(image)[0][0]), cmap='gray')
            ax += 1
            print(output.shape)
            for filter in filters[l]:
                axes[ax].pcolormesh(np.flipud(output[0][filter]), cmap='gray')
                ax += 1