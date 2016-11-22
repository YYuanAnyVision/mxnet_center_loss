import mxnet as mx
import numpy as np
from center_loss import *
from data import mnist_iterator
import logging
import train_model
import argparse

parser = argparse.ArgumentParser(description='train mnist use softmax and centerloss')
parser.add_argument('--gpus', type=str, default='',
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--batch-size', type=int, default=100,
                    help='the batch size')
parser.add_argument('--num-examples', type=int, default=60000,
                    help='the number of training examples')
parser.add_argument('--lr', type=float, default=.01,
                    help='the initial learning rate')
parser.add_argument('--lr-factor', type=float, default=0.5,
                    help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add_argument('--lr-factor-epoch', type=float, default=20,
                    help='the number of epoch to factor the lr, could be .5')
parser.add_argument('--model-prefix', type=str,
                    help='the prefix of the model to load')
parser.add_argument('--save-model-prefix', type=str,default='center_loss',
                    help='the prefix of the model to save')
parser.add_argument('--num-epochs', type=int, default=20,
                    help='the number of training epochs')
parser.add_argument('--load-epoch', type=int,
                    help="load the model on an epoch using the model-prefix")
parser.add_argument('--kv-store', type=str, default='local',
                    help='the kvstore type')
parser.add_argument('--log_file', type=str, default='log.txt',
                    help='log file')
parser.add_argument('--log_dir', type=str, default='.',
                    help='log dir')
args = parser.parse_args()

# mnist input shape
data_shape = (1,28,28)

def get_symbol(batchsize=64):
    """
    LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
    Haffner. "Gradient-based learning applied to document recognition."
    Proceedings of the IEEE (1998)
    """
    data = mx.symbol.Variable('data')

    softmax_label = mx.symbol.Variable('softmax_label')
    center_label = mx.symbol.Variable('center_label')

    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",kernel=(2,2), stride=(2,2))

    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",kernel=(2,2), stride=(2,2))

    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")

    embedding = mx.symbol.FullyConnected(data=tanh3, num_hidden=2, name='embedding')

    # second fullc
    fc2 = mx.symbol.FullyConnected(data=embedding, num_hidden=10, name='fc2')

    ce_loss = mx.symbol.SoftmaxOutput(data=fc2, label=softmax_label, name='softmax')

    center_loss_ = mx.symbol.Custom(data=embedding, label=center_label, name='center_loss_', op_type='centerloss',\
            num_class=10, alpha=0.5, scale=1.0, batchsize=batchsize)
    center_loss = mx.symbol.MakeLoss(name='center_loss', data=center_loss_)
    mlp = mx.symbol.Group([ce_loss, center_loss])

    return mlp

def main():
    batchsize = args.batch_size if args.gpus is '' else \
        args.batch_size / len(args.gpus.split(','))
    print 'batchsize is ', batchsize

    # define network structure
    net = get_symbol(batchsize)

    # load data
    train, val = mnist_iterator(batch_size=args.batch_size, input_shape=data_shape)

    # train
    print 'training model ...'
    train_model.fit(args, net, (train, val), data_shape)

if __name__ == "__main__":
    main()
