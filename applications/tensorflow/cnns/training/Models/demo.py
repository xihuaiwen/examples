import tensorflow as tf
from functools import partial

cfg={
    'conv1':      {'out_filters':32,'ksize':3,'strides':2},
    'conv2':      {'out_filters':64,'ksize':3,'strides':2},
    'maxpool1':   {'ksize':2,'strides':2},
    'fc':         {'out_filters':10}
}

def conv(x,out_filters,ksize,strides=1,name=None):
    with tf.variable_scope(name,use_resource=True, reuse=tf.AUTO_REUSE):
        x = tf.layers.conv2d(inputs=x, filters=out_filters, kernel_size=ksize,activation=tf.nn.relu6,name=name)
        return x
def fc(x,out_filters,name=None):
    with tf.variable_scope('flat',use_resource=True, reuse=tf.AUTO_REUSE):
        x = tf.layers.flatten(x)
    with tf.variable_scope(name,use_resource=True, reuse=tf.AUTO_REUSE):
        x = tf.layers.dense(x,units=out_filters)
        return x
def pool(x,ksize,strides,name=None):
    with tf.variable_scope(name,use_resource=True, reuse=tf.AUTO_REUSE):
        x = tf.layers.max_pooling2d(x,pool_size=2,strides=2)
        return x
        
class Demo:
    def __init__(self, opts, is_training=True):
        self.is_training = is_training
        
    def __call__(self,x):
        fn_list = self._build_function_list()
        tf.add_to_collection("activations", x)
        with tf.variable_scope("all", use_resource=True):
            for fn in fn_list:
                x = fn(x)
            return x        
        
    def _build_function_list(self):
        fn_list = []
        for key,val in cfg.items():
            if 'conv' in key:
                fn_list.append(partial(conv,out_filters=val['out_filters'],ksize=val['ksize'],strides=val['strides'],name=key))
            elif 'pool' in key:
                fn_list.append(partial(pool,ksize=val['ksize'],strides=val['strides'],name=key))
            elif 'fc' in key:
                fn_list.append(partial(fc,out_filters=val['out_filters'],name=key))
        return fn_list
    
def Model(opts, training, image):
    return Demo(opts, training)(image)

def set_defaults(opts):
    
    if not opts.get("learning_rate_decay"):
        opts["learning_rate_decay"] = [1.0, 0.1, 0.01]  
    if not opts.get("base_learning_rate"):
        opts["base_learning_rate"] = -6  

    if not opts.get('epochs') and not opts.get('iterations'):
        opts['epochs'] = 100
    if opts.get("group_norm"):
        if not opts.get("groups"):
            opts["groups"] = 32
    if not opts.get("batch_size"):
        opts['batch_size'] = 4
         
    if not opts.get("model_size"):
        opts['model_size'] = 'V1'     
         
    if (opts['precision'] == '32.32') and not opts.get("shards"):
        opts['shards'] = 2
         
    opts['name'] = "SN_bs{}".format(opts['batch_size'])
    if opts.get('replicas') > 1:
        opts['name'] += "x{}r".format(opts['replicas'])
    if opts['pipeline_depth'] > 1:
        opts['name'] += "x{}p".format(opts['pipeline_depth'])
    elif opts.get('gradients_to_accumulate') > 1:
        opts['name'] += "x{}a".format(opts['gradients_to_accumulate'])    
    return opts

def add_arguments(parser):
    return parser