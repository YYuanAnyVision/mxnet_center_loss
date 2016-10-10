import os
import mxnet as mx
import numpy as np
from data import mnist_iterator
import logging
from center_loss import *
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

def visual_feature_space(features,labels, num_classes, name_dict):
    num = len(labels)

    # draw
    palette = np.array(sns.color_palette("hls", num_classes))
    
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(features[:,0], features[:,1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    ax.axis('off')
    ax.axis('tight')
    
    # We add the labels for each digit.
    txts = []
    for i in range(num_classes):
        # Position of each label.
        xtext, ytext = np.median(features[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, name_dict[i])
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.show()
    return f, ax, sc, txts

def main():
    # load model, get embedding layer
    model = mx.model.FeedForward.load('center_loss', 20, ctx=mx.cpu(0), numpy_batch_size=1)
    internals = model.symbol.get_internals()
    embedding_layer = internals['embedding_output']
    feature_extractor = mx.model.FeedForward(ctx=mx.cpu(0), symbol=embedding_layer, numpy_batch_size=1,\
            arg_params = model.arg_params, aux_params=model.aux_params, allow_extra_params=True)
    print 'feature_extractor loaded'

    # load MNIST data 
    _, val = mnist_iterator(batch_size=100, input_shape=(1,28,28))
    
    # extract feature 
    print 'extracting feature'
    embeds = []
    labels = []
    for i in val:
        preds = feature_extractor.predict( i.data[0] )
        embeds.append( preds )
        labels.append( i.label[0].asnumpy())

    embeds = np.vstack(embeds)
    labels = np.hstack(labels)

    print 'embeds shape is ', embeds.shape
    print 'labels shape is ', labels.shape

    # prepare dict for display
    namedict = dict()
    for i in range(10):
        namedict[i]=str(i)

    visual_feature_space(embeds, labels, 10, namedict)

if __name__ == "__main__":
    main()

