import tensorflow as tf
from functools import partial

'''
two loop for 'for' ops
bwd_inputs = [t for op in grad_ops for t in op.inputs]
'''

vgg_config={
    '11-A':{'layers':[1,1,2,2,2],'filters':[]},
    '11-ALRN':{'layers':[1,1,2,2,2],'filters':[]},
    '13-B':{'layers':[2,2,3,3,3],'filters':[]},
    '16-C':{'layers':[2,2,2,2,2],'filters':[64,128,256,512,512],'use1x1':True},
    '16-D':{'layers':[2,2,3,3,3],'filters':[64,128,256,512,512],'use1x1':True},
    '19-E':{'layers':[2,2,4,4,4],'filters':[]},
}


class VggNet:
    def __init__(self, opts, is_training=True):
        self.is_training = is_training
        self.num_classes = 1000
        self.batch_size = opts['batch_size']
        self.backbone = '16-D'

        dtypes = opts["precision"].split(".")
        self.dtype = tf.float16 if dtypes[0] == "16" else tf.float32

        self.master_weight_filter_fn = (
            lambda name: tf.float32 if dtypes[1] == "32" else tf.float16
        )

        self.custom_dtype_getter = partial(
            custom_dtype_getter,
            master_weight_filter_fn=self.master_weight_filter_fn,
        )
    
    def __call__(self, x):
        fn_list = self._build_function_list()

        tf.add_to_collection("activations", x)
        with tf.variable_scope(
            "all", use_resource=True, custom_getter=self.custom_dtype_getter
        ):
            for fn in fn_list:
                x = fn(x)
        return x
        
    def _build_function_list(self):
        fn_list = []

        for i in range(len(vgg_config[self.backbone]['layers'])):
            for j in range(vgg_config[self.backbone]['layers'][i]):
                fn_list.append(partial(
                _conv_layer,
                filters=vgg_config[self.backbone]['filters'][i],
                name="b{}/{}/conv".format(i, j)))
            fn_list.append(partial(
                _max_pool,
                name="b{}/pool".format(i)
            ))
        
        # fn_list.append(partial(
        #     tf.reshape,
        #     shape=(self.batch_size,-1),
        #     name='reshape'
        # ))
        for index,k in enumerate([4096,1000]):
            fn_list.append(
                partial(
                    self._fc_layer,
                    out=k,
                    name='fc'+str(6+index)
                )
            )
        return fn_list

    def first_stage(self, x, first_split_name):
        self.fn_list = self._build_function_list()
        if first_split_name not in [f.keywords["name"] for f in self.fn_list]:
            raise ValueError(
                "Couldn't find pipeline split called " + first_split_name
            )
        tf.add_to_collection("activations", x)
        with tf.variable_scope(
            "all", use_resource=True, custom_getter=self.custom_dtype_getter
        ):
            for fn in self.fn_list:
                if fn.keywords["name"] == first_split_name:
                    break
                x = fn(x)
        return x

    def later_stage(self, x, prev_split_name, end_split_name):
        if end_split_name is not None and end_split_name not in [
            fn.keywords["name"] for fn in self.fn_list
        ]:
            raise ValueError(
                "Couldn't find pipeline split called " + end_split_name
            )
        with tf.variable_scope(
            "all", use_resource=True, custom_getter=self.custom_dtype_getter
        ):
            first_stage = False
            for f in self.fn_list:
                if (not first_stage and f.keywords["name"] != prev_split_name):
                    continue
                first_stage = True
                if f.keywords["name"] == end_split_name:
                    break
                x = f(x)
        return x

    def _fc_layer(self,x, out, name):
        # with tf.variable_scope(name, use_resource=True):
        #     x = tf.layers.dense(x, out)
        #     if out==1000:
        #         x = tf.nn.softmax(x, name="prob")
        #     else:
        #         x = tf.nn.relu(x)
        #         if self.is_training:
        #             x = tf.nn.dropout(x, 0.5)
        #     return x
        in_filters = x.get_shape().as_list()[3]
        with tf.variable_scope(name, use_resource=True):
            if out==1000:
                x = tf.nn.conv2d(x,filters=[1,1,in_filters,out])
                # x = tf.layers.batch_normalization()
                x = tf.nn.softmax(x, name="prob")
            else:
                ff = tf.Tensor(op, value_index, dtype)
                x = tf.nn.atrous_conv2d(x,filters=[3,3,in_filters,out],rate=2)
                x = tf.nn.relu(x)

def _block_n(x,out_filters,layers,name):
    with tf.variable_scope(name, use_resource=True):
        for i in range(layers):
            x = _conv_layer(x,out_filters,name=name+"conv_"+str(i))
        x = _max_pool(x, name=name+"pool")
        return x



def _conv_layer(x, filters,stride=1, kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), ksize=3, bias=True, name=None):
    with tf.variable_scope(name, use_resource=True):
        return tf.layers.conv2d(
            inputs=x, filters=filters, kernel_size=ksize, strides=stride,
            padding='same',
            use_bias=bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002),
            activation=tf.nn.relu,
            data_format='channels_last')


def _max_pool(x, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], name=None):
    with tf.variable_scope(name, use_resource=True):
        return tf.nn.max_pool(x, ksize, stride,
                            padding='SAME')


def custom_dtype_getter(getter, name, dtype, trainable,
                        master_weight_filter_fn,
                        shape=None, *args, **kwargs):
    master_dtype = master_weight_filter_fn(name)
    if dtype != master_dtype and trainable:
        var = getter(
            name, shape, master_dtype, *args, trainable=trainable, **kwargs
        )
        return tf.cast(var, dtype=dtype, name=name + "_cast")
    else:
        return getter(name, shape, dtype, *args, trainable=trainable, **kwargs)

def Model(opts, training, image):
    return VggNet(opts, training)(image)


def staged_model(opts):
    splits = opts['pipeline_splits']
    x = VggNet(opts, True)
    if splits is None:
        possible_splits = [
            s.keywords['name'] for s in x._build_function_list()
            if 'conv' or 'pool' in s.keywords['name']
        ]
        raise ValueError(
            "--pipeline-splits not specified. Need {} of {}".format(
                opts['shards'] - 1, possible_splits))
    splits.append(None)
    stages = [partial(x.first_stage, first_split_name=splits[0])]
    for i in range(len(splits) - 1):
        stages.append(
            partial(x.later_stage,
                    prev_split_name=splits[i],
                    end_split_name=splits[i + 1]))
    return stages

def add_arguments(parser):
    return parser

def set_defaults(opts):
    opts['summary_str'] += "VGG16\n"

    # if opts['dataset'] == 'imagenet':
    #     opts['shortcut_type'] = 'B'
    # elif 'cifar' in opts['dataset']:
    #     opts['shortcut_type'] = 'A'

#    opts['dataset'] = 'imagenet'
    opts['lr_schedule'] = 'polynomial_decay_lr'

    if not opts.get("learning_rate_decay"):
        opts["learning_rate_decay"] = [1.0, 0.1, 0.01]  
    if not opts.get("base_learning_rate"):
        opts["base_learning_rate"] = -6  

    if not opts.get('epochs') and not opts.get('iterations'):
        opts['epochs'] = 100

    if not opts.get("batch_size"):
        opts['batch_size'] = 4

    if (opts['precision'] == '32.32') and not opts.get("shards"):
        opts['shards'] = 2

    opts['name'] = "SN_bs{}".format(opts['batch_size'])

    if opts.get('replicas') > 1:
        opts['name'] += "x{}r".format(opts['replicas'])
    if opts['pipeline_depth'] > 1:
        opts['name'] += "x{}p".format(opts['pipeline_depth'])
    elif opts.get('gradients_to_accumulate') > 1:
        opts['name'] += "x{}a".format(opts['gradients_to_accumulate'])

    opts['name'] += '_{}{}'.format(opts['precision'],
                                   '_noSR' if opts['no_stochastic_rounding'] else '')