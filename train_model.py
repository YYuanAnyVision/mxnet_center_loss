import mxnet as mx
import logging
import os
from center_loss import *


def get_model_dict(network, data_shape):
    '''
        return the (name,shape) dict for both args and aux,
        so that in the finetune process, new model will only load
        those valid params
    '''
    arg_shapes, output_shapes, aux_shapes = network.infer_shape( data=(1,)+data_shape )
    arg_names = network.list_arguments()
    aux_names = network.list_auxiliary_states()

    arg_dict = dict(zip(arg_names, arg_shapes))
    aux_dict = dict(zip(aux_names, aux_shapes))
    return arg_dict, aux_dict

def fit(args, network, data_loader, data_shape, batch_end_callback=None, patterns=None, initializers=None):
    # kvstore
    kv = mx.kvstore.create(args.kv_store)
    
    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    if 'log_file' in args and args.log_file is not None:
        log_file = args.log_file
        log_dir = args.log_dir
        log_file_full_name = os.path.join(log_dir, log_file)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        logger = logging.getLogger()
        handler = logging.FileHandler(log_file_full_name)
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info('start with arguments %s', args)
    else:
        logging.basicConfig(level=logging.DEBUG, format=head)
        logging.info('start with arguments %s', args)

    # load model
    model_prefix = args.model_prefix
    model_args = {}
    if args.load_epoch is not None:
        assert model_prefix is not None
        tmp = mx.model.FeedForward.load(model_prefix, args.load_epoch)
        
        # only add those with the same shape
        arg_dict, aux_dict = get_model_dict( network, data_shape )
        valid_arg = dict()
        valid_aux = dict()
        
        # print all the parameters
        print 'all params ', arg_dict

        # for args 
        for k, v in arg_dict.items():
            # skip those 'label'
            if k == 'data' or k.endswith('label'):
                continue

            # skip those pretrain model dosen't have
            if not k in tmp.arg_params.keys():
                continue

            if v == tmp.arg_params[k].shape:
                valid_arg[k] = tmp.arg_params[k]
                print 'catching arg: {} from pretrained model'.format(k)
        # for aux 
        for k, v in aux_dict.items():
            # skip these 'label'
            if k == 'data' or k.endswith('label'):
                continue
            
            # skip those pretrain model dosen't have
            if not k in tmp.aux_params.keys():
                continue

            if v == tmp.aux_params[k].shape:
                valid_aux[k] = tmp.aux_params[k]
                print 'catching aux: {} from pretrained model'.format(k)

        model_args = {'arg_params' : valid_arg,
                      'aux_params' : valid_aux,
                      'begin_epoch' : args.load_epoch}
    # save model
    save_model_prefix = args.save_model_prefix
    if save_model_prefix is None:
        save_model_prefix = model_prefix
    checkpoint = None if save_model_prefix is None else mx.callback.do_checkpoint(save_model_prefix)

    # data
    (train, val) = data_loader

    # train
    devs = mx.cpu() if args.gpus is '' else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    epoch_size = args.num_examples / args.batch_size

    if args.kv_store == 'dist_sync':
        epoch_size /= kv.num_workers
        model_args['epoch_size'] = epoch_size

    if 'lr_factor' in args and args.lr_factor < 1:
        model_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
            step = max(int(epoch_size * args.lr_factor_epoch), 1),
            factor = args.lr_factor)

    if 'clip_gradient' in args and args.clip_gradient is not None:
        model_args['clip_gradient'] = args.clip_gradient

    # disable kvstore for single device
    if 'local' in kv.type and (
            args.gpus is None or len(args.gpus.split(',')) is 1):
        kv = None
    
    init_patterns = ['.*fc.*', '.*']
    init_methods = [ mx.init.Normal(sigma=0.001), mx.init.Xavier(factor_type="out", rnd_type="gaussian", magnitude=2.0)]

    print 'dev is ',devs
    model = mx.model.FeedForward(
        ctx                = devs,
        symbol             = network,
        num_epoch          = args.num_epochs,
        learning_rate      = args.lr,
        momentum           = 0.9,
        wd                 = 0.0005,
        initializer        = mx.init.Mixed(init_patterns, init_methods),
        **model_args)

    if batch_end_callback is not None:
        if not isinstance(batch_end_callback, list):
            batch_end_callback = [batch_end_callback]
    else:
        batch_end_callback = []
    batch_end_callback.append(mx.callback.Speedometer(args.batch_size, 50))

    # custom metric
    eval_metrics = mx.metric.CompositeEvalMetric()
    eval_metrics.add(Accuracy())
    eval_metrics.add(CenterLossMetric())
    
    model.fit(
        X                  = train,
        eval_metric        = eval_metrics,
        eval_data          = val,
        kvstore            = kv,
        batch_end_callback = batch_end_callback,
        epoch_end_callback = checkpoint)
