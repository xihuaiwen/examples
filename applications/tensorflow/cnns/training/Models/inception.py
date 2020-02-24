import tensorflow as tf
from functools import partial
from tensorflow.python.ipu import normalization_ops

cfg={
    'V1':{'3a':[64,96,128,16,32,32],
          '3b':[128,128,192,32,96,64],
          '4a':[192,96,208,16,48,64],
          '4b':[160,112,224,24,64,64],
          '4c':[128,128,256,24,64,64],
          '4d':[112,144,288,32,64,64],
          '4e':[256,160,320,32,128,128],
          '5a':[256,160,320,32,128,128],
          '5b':[384,192,384,48,128,128]},    
    'V2':{'layers':[1,1,2,2,2],'filters':[]},
    'V3':{'layers':[2,2,3,3,3],'filters':[]},
    'V4':{'layers':[2,2,2,2,2],'filters':[]},
    'VR':{'layers':[2,2,3,3,3],'filters':[]},
}


def _conv_layer(x, filters_out,stride=1, kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), ksize=3, groups=1,bias=True, bn_cfg=None, seed=None, name=None):
     in_filters = x.get_shape().as_list()[3]
     with tf.variable_scope(name, use_resource=True, reuse=tf.AUTO_REUSE):
          if stride > 1:
               x = _fixed_padding(x, ksize, "channels_last")
          W = tf.get_variable(
               "conv2d/kernel",
               shape=[ksize, ksize, in_filters / groups, filters_out],
               dtype=x.dtype,
               trainable=True,
               initializer=tf.variance_scaling_initializer(seed=seed),
          )
          x = tf.nn.conv2d(
            x,
            filters=W,
            strides=[1, stride, stride, 1],
            padding=("SAME" if stride == 1 else "VALID"),
            data_format="NHWC",
          )
          if bn_cfg is not None:
               x = _norm(x,bn_cfg)
          return tf.nn.relu(x)
     
def _fixed_padding(inputs, kernel_size, data_format):
     """Pads the input along the spatial dimensions independently of input size.

     Further details of this is necessary can be found at:
     https://www.tensorflow.org/versions/r1.8/api_guides/python/nn#Convolution
     """
     pad_total = kernel_size - 1
     pad_beg = pad_total // 2
     pad_end = pad_total - pad_beg

     if data_format == "channels_first":
          padded_inputs = tf.pad(
             inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]]
        )
     else:
          padded_inputs = tf.pad(
             inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
        )
     return padded_inputs

def _pool(x, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], model='max', name=None):
     with tf.variable_scope(model+name, use_resource=True):
          if model is 'max':
               x = tf.nn.max_pool(x, ksize, stride,padding='SAME')
          elif model is 'avg':
               x = tf.nn.avg_pool(x,ksize,stride,padding='VALID')
          else:
               NotImplemented
     return x
     
def _norm(x, opts, is_training=True):
     norm_type = (
         "GROUP"
        if opts["group_norm"]
        else "BATCH"
        if opts["batch_norm"]
        else None
    )

     if norm_type == "BATCH":
          x = tf.layers.batch_normalization(
             x,
            fused=True,
            center=True,
            scale=True,
            training=is_training,
            trainable=True,
            momentum=opts["BN_decay"],
            epsilon=1e-5,
        )
     elif norm_type == "GROUP":
          x = normalization_ops.group_norm(x, groups=opts["groups"])

     tf.add_to_collection("activations", x)
     return x

class Inception:
     def __init__(self, opts, is_training=True):
          self.is_training = is_training
          self.num_classes = 1000
          self.opts = opts
          self.model = opts['model_size']
          dtypes = opts["precision"].split(".")
          self.dtype = tf.float16 if dtypes[0] == "16" else tf.float32
  
          self.master_weight_filter_fn = (
              lambda name: tf.float32 if dtypes[1] == "32" else tf.float16
          )
  
          self.custom_dtype_getter = partial(
              custom_dtype_getter,
              master_weight_filter_fn=self.master_weight_filter_fn,
          )          
          pass
     
     def __call__(self, x):
          result = []
          fn_list = self._build_function_list()

          tf.add_to_collection("activations", x)
          #with tf.variable_scope(
             #"all", use_resource=True, custom_getter=self.custom_dtype_getter):
          for each_fn in fn_list:
               ret = x
               for fn in each_fn:
                    ret = fn(ret)
               result.append(ret)
          return result
     
     def _block(self,x,out_filters,model_session,name=None):
          with tf.variable_scope(name):
               branch_0 = _conv_layer(x,filters_out=cfg[self.model][model_session][0],ksize=1,bn_cfg=self.opts,name='branch_0')
               branch_1 = _conv_layer(x,filters_out=cfg[self.model][model_session][1],ksize=1,bn_cfg=self.opts,name='branch_1_0')
               branch_1 = _conv_layer(branch_1,filters_out=cfg[self.model][model_session][2],ksize=3,bn_cfg=self.opts,name='branch_1_1')
               branch_2 = _conv_layer(x,filters_out=cfg[self.model][model_session][3],ksize=1,bn_cfg=self.opts,name='branch_2_0')
               branch_2 = _conv_layer(branch_2,filters_out=cfg[self.model][model_session][4],ksize=3,bn_cfg=self.opts,name='branch_2_1') 
               branch_3 = _pool(x, ksize=3,stride=1,model='max',name='branch_3_0')
               branch_3 = _conv_layer(branch_3,filters_out=cfg[self.model][model_session][5],ksize=1,bn_cfg=self.opts,name='branch_3_1')      
               return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
     
     def _init_block(self,x,out_filters,name=None):
          with tf.variable_scope(name):
               x = _conv_layer(x,filters_out=out_filters,ksize=7,stride=2,bn_cfg=self.opts,name='conv_0')
               x = _pool(x,ksize=3,stride=2,model='max',name='pool_0')
               x = _conv_layer(x,filters_out=64,ksize=1,stride=1,bn_cfg=self.opts,name='conv_1')
               x = _conv_layer(x,filters_out=192,ksize=3,stride=1,bn_cfg=self.opts,name='conv_2')
               x = _pool(x,ksize=3,stride=2,model='max',name='pool_1')
               return x
     
     def _final_block(self,x,name=None):
          with tf.variable_scope(name):
               x = tf.layers.average_pooling2d(x, pool_size=7, strides=1, name="final_pool") 
               #x = _conv_layer(x,filters_out=self.num_classes,ksize=1,stride=1,bn_cfg=self.opts,name='conv_3')
               x = tf.nn.dropout(x, keep_prob=0.5 if self.is_training else 0.0, name="drop")
               x = tf.layers.flatten(x)
               return x               

     #def _extra_block(self,x,name=None):
          #with tf.variable_scope(name):
               #x = tf.layers.average_pooling2d(x, pool_size=7, strides=8, name="extra_pool") 
               ##filters = tf.get_variable(
                    ##"conv2d/extra_block",
                    ##shape=[2,2,x.get_shape().as_list()[3],self.num_classes],
                    ##dtype=x.dtype,
                    ##trainable=True,
                    ##initializer=tf.variance_scaling_initializer(seed=None),
               ##)               
               ##x = tf.nn.conv2d(x,filters=filters,strides=1,padding="VALID",name='conv_3')
               #x = _conv_layer(x,filters_out=self.num_classes,ksize=1,stride=1,bn_cfg=self.opts,name='conv_4')
               ##x = tf.layers.average_pooling2d(x, pool_size=5, strides=1, name="extra_pool")
               #x = tf.nn.dropout(x, keep_prob=0.5 if self.is_training else 0.0, name="drop")
               #x = tf.layers.flatten(x)
               ##x = tf.layers.dense(x, units=self.num_classes, activation=None)
               #return x
          
     def _build_function_list(self):
          all_fn_list = []
          #ext1_fn_list = []
          #ext2_fn_list =[]
          
          all_fn_list.append(partial(self._init_block,out_filters=64,name='init_block'))
          for i,key in enumerate(cfg[self.model]):
               all_fn_list.append(partial(self._block,out_filters=cfg[self.model][key],model_session=key,name='block_{}'.format(i+1)))
               #if key is '4a':
                    #ext1_fn_list = all_fn_list.copy()
                    #ext1_fn_list.append(partial(self._extra_block,name='extra_block1'))
               #elif key is '4d':
                    #ext2_fn_list = all_fn_list.copy()
                    #ext2_fn_list.append(partial(self._extra_block,name='extra_block2'))
               if key in ['3b','4e']:
                    all_fn_list.append(partial(_pool,ksize=3,stride=2,model='max',name='pool_{}'.format(i+1)))
          all_fn_list.append(partial(self._final_block,name='final_block'))
          
          return [all_fn_list]
          
          
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
     return Inception(opts, training)(image)


def staged_model(opts):
     splits = opts['pipeline_splits']
     x = Inception(opts, True)
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
     parser.add_argument("--group-norm", action="store_true",
                        help="Use group norm (ImageNet Default)")
     parser.add_argument("--groups", type=int, help="Number of groups")     
     parser.add_argument("--batch-norm", action="store_true",
                        help="Use batch norm (CIFAR Default)")     
     return parser

def set_defaults(opts):
     opts['summary_str'] += "Inception\n"

#    opts['dataset'] = 'imagenet'
     #opts['lr_schedule'] = 'polynomial_decay_lr'

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

     opts['name'] += '_{}{}'.format(opts['precision'],
                                   '_noSR' if opts['no_stochastic_rounding'] else '')
               
          