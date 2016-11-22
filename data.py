# pylint: skip-file
""" data iterator for mnist """
import sys
import os

# code to automatically download dataset
mxnet_root = ''
sys.path.append(os.path.join( mxnet_root, 'tests/python/common'))
import get_data
import mxnet as mx


class custom_mnist_iter(mx.io.DataIter):
    def __init__(self, mnist_iter):
        super(custom_mnist_iter,self).__init__()
        self.data_iter = mnist_iter
        self.batch_size = self.data_iter.batch_size

    @property
    def provide_data(self):
        return self.data_iter.provide_data

    @property
    def provide_label(self):
        provide_label = self.data_iter.provide_label[0]
        return [('softmax_label', provide_label[1]), \
                ('center_label', provide_label[1])]

    def hard_reset(self):
        self.data_iter.hard_reset()

    def reset(self):
        self.data_iter.reset()

    def next(self):
        batch = self.data_iter.next()
        label = batch.label[0]

        return mx.io.DataBatch(data=batch.data, label=[label,label], \
                pad=batch.pad, index=batch.index)
    


def mnist_iterator(batch_size, input_shape):
    """return train and val iterators for mnist"""
    # download data
    get_data.GetMNIST_ubyte()
    flat = False if len(input_shape) == 3 else True

    train_dataiter = mx.io.MNISTIter(
        image="data/train-images-idx3-ubyte",
        label="data/train-labels-idx1-ubyte",
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=True,
        flat=flat)

    val_dataiter = mx.io.MNISTIter(
        image="data/t10k-images-idx3-ubyte",
        label="data/t10k-labels-idx1-ubyte",
        input_shape=input_shape,
        batch_size=batch_size,
        flat=flat)

    return (custom_mnist_iter(train_dataiter), custom_mnist_iter(val_dataiter))
