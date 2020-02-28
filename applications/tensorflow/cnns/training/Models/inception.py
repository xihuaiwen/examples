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
    'V3':{'block1':{'module1':[64,64,48,96,96,64,32,256],
                    'module2':[64,64,48,96,96,64,64,288],
                    'module3':[64,64,48,96,96,64,64,288]},
          'block2':{'module1':[384,96,96,64,768],
                    'module2':[192,192,128,128,192,128,128,128,128,192,768],
                    'module3':[192,192,160,160,192,160,160,160,160,192,768],
                    'module4':[192,192,160,160,192,160,160,160,160,192,768],
                    'module5':[192,192,192,192,192,192,192,192,192,192,768]},
          'block3':{'module1':[320,192,192,192,192,192,1280],
                    'module2':[320,384,384,384,448,384,384,384,192,2048]}},
    'V4':{'layers':[2,2,2,2,2],'filters':[]},
    'VR':{'layers':[2,2,3,3,3],'filters':[]},
}


def _conv(x, filters_out,stride=1, ksize=3, groups=1,bias=True, bn_cfg=None, seed=None, padding='valid', name=None):
     with tf.variable_scope(name,use_resource=True, reuse=tf.AUTO_REUSE):
          x = tf.layers.conv2d(inputs=x, filters=filters_out, kernel_size=ksize, kernel_initializer=tf.contrib.layers.xavier_initializer(),strides=stride, activation=None, padding=padding,name=name)
          x = _norm(x,bn_cfg)
          return tf.nn.relu(x)         

def _pool(x, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], model='max', padding='SAME', name=None):
     with tf.variable_scope(model+name, use_resource=True):
          if model is 'max':
               x = tf.nn.max_pool(x, ksize, stride,padding=padding)
          elif model is 'avg':
               x = tf.nn.avg_pool(x,ksize,stride,padding=padding)
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

     # tf.add_to_collection("activations", x)
     return x

class Inception:
     def __init__(self, opts, is_training=True):
          self.is_training = is_training
          self.num_classes = 1000
          self.opts = opts
          self.model = opts['model_version']
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
             "all", use_resource=True, custom_getter=self.custom_dtype_getter):
               for fn in fn_list:
                    x = fn(x)
               return x
          
     def _block_v1(self,x,out_filters,name=None):
          with tf.variable_scope(name):
               branch_0 = _conv(x,filters_out=out_filters[0],ksize=1,bn_cfg=self.opts,padding='same',name='branch_0')
               branch_1 = _conv(x,filters_out=out_filters[1],ksize=1,bn_cfg=self.opts,padding='same',name='branch_1_0')
               branch_1 = _conv(branch_1,filters_out=out_filters[2],ksize=3,bn_cfg=self.opts,padding='same',name='branch_1_1')
               branch_2 = _conv(x,filters_out=out_filters[3],ksize=1,bn_cfg=self.opts,padding='same',name='branch_2_0')
               branch_2 = _conv(branch_2,filters_out=out_filters[4],ksize=3,bn_cfg=self.opts,padding='same',name='branch_2_1') 
               branch_3 = _pool(x, ksize=3,stride=1,model='max',padding='SAME',name='branch_3_0')
               branch_3 = _conv(branch_3,filters_out=out_filters[5],ksize=1,bn_cfg=self.opts,padding='same',name='branch_3_1')      
               return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
     
     def _init_block(self,x,out_filters,name=None):
          with tf.variable_scope(name):
               if self.opts['model_version'] in 'V1':
                    x = _conv(x,filters_out=out_filters,ksize=7,stride=2,bn_cfg=self.opts,padding='same',name='conv_0')
                    x = _pool(x,ksize=3,stride=2,model='max',padding='SAME',name='pool_0')
                    x = _conv(x,filters_out=192,ksize=3,stride=1,bn_cfg=self.opts,padding='same',name='conv_1')
                    #x = _conv_layer(x,filters_out=192,ksize=3,stride=1,bn_cfg=self.opts,padding='same',name='conv_2')
                    x = _pool(x,ksize=3,stride=2,model='max',padding='SAME',name='pool_1')
               elif self.opts['model_version'] in 'V3':
                    x = _conv(x,filters_out=out_filters,ksize=3,stride=2,bn_cfg=self.opts,padding='same',name='conv_0')
                    x = _conv(x,filters_out=out_filters,ksize=3,stride=1,bn_cfg=self.opts,padding='valid',name='conv_1')
                    x = _conv(x,filters_out=64,ksize=3,stride=1,bn_cfg=self.opts,padding='same',name='conv_2')
                    x = _pool(x,ksize=3,stride=2,model='max',padding='SAME',name='pool_0')
                    x = _conv(x,filters_out=80,ksize=3,stride=1,bn_cfg=self.opts,padding='valid',name='conv_3')
                    x = _conv(x,filters_out=192,ksize=3,stride=2,bn_cfg=self.opts,padding='same',name='conv_4')
                    x = _conv(x,filters_out=288,ksize=3,stride=1,bn_cfg=self.opts,padding='same',name='conv_5')                    
               return x
     
     def _final_block(self,x,ksize,name=None):
          with tf.variable_scope(name):
               x = _pool(x, ksize=ksize, stride=1,model='max',padding='VALID',name="pool_2") 
               x = tf.nn.dropout(x, keep_prob=0.5 if self.is_training else 0.0, name="drop")
               x = tf.reduce_mean(x,reduction_indices=[1, 2])
               x = tf.layers.dense(x, units=self.num_classes,kernel_initializer=tf.glorot_uniform_initializer())
               return x  
          
     def _block_v3_1(self,x,out_filters,name=None):
          with tf.variable_scope(name):
               #1x1
               branch_0 = _conv(x,filters_out=out_filters[0],ksize=1,bn_cfg=self.opts,padding='same',name='branch_0')
               #1x1 > 3x3
               branch_1 = _conv(x,filters_out=out_filters[1],ksize=1,bn_cfg=self.opts,padding='same',name='branch_1_0')
               branch_1 = _conv(branch_1,filters_out=out_filters[2],ksize=5,bn_cfg=self.opts,padding='same',name='branch_1_1')
               #1x1 > 3x3 > 3x3
               branch_2 = _conv(x,filters_out=out_filters[3],ksize=1,bn_cfg=self.opts,padding='same',name='branch_2_0')
               branch_2 = _conv(branch_2,filters_out=out_filters[4],ksize=3,bn_cfg=self.opts,padding='same',name='branch_2_1')
               branch_2 = _conv(branch_2,filters_out=out_filters[5],ksize=3,bn_cfg=self.opts,padding='same',name='branch_2_2')               
               #pool > 1x1
               branch_3 = _pool(x, ksize=3,stride=1,model='avg',padding='SAME',name='branch_3_0')
               branch_3 = _conv(branch_3,filters_out=out_filters[6],ksize=1,bn_cfg=self.opts,padding='same',name='branch_3_1')     
               return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
          
     def _block_v3_2(self,x,out_filters,name=None):
          with tf.variable_scope(name):
               if 'module1' in name:
                    #3x3
                    branch_0 = _conv(x,filters_out=out_filters[0],ksize=3,stride=2,bn_cfg=self.opts,padding='same',name='branch_0')
                    #1x1 > 3x3 > 3x3
                    branch_1 = _conv(x,filters_out=out_filters[1],ksize=1,bn_cfg=self.opts,padding='same',name='branch_1_0')
                    branch_1 = _conv(branch_1,filters_out=out_filters[2],ksize=3,bn_cfg=self.opts,padding='same',name='branch_1_1')
                    branch_1 = _conv(branch_1,filters_out=out_filters[3],ksize=3,stride=2,bn_cfg=self.opts,padding='same',name='branch_1_2')    
                    #pool
                    branch_2 = _pool(x, ksize=3,stride=2,model='max',padding='SAME',name='branch_2')     
                    return tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
               else:
                    #1x1
                    branch_0 = _conv(x,filters_out=out_filters[0],ksize=1,bn_cfg=self.opts,padding='same',name='branch_0')
                    #1x1 > 1x7 > 7x1
                    branch_1 = _conv(x,filters_out=out_filters[1],ksize=1,bn_cfg=self.opts,padding='same',name='branch_1_0')
                    branch_1 = _conv(branch_1,filters_out=out_filters[2],ksize=[1,7],bn_cfg=self.opts,padding='same',name='branch_1_1')
                    branch_1 = _conv(branch_1,filters_out=out_filters[3],ksize=[7,1],bn_cfg=self.opts,padding='same',name='branch_1_2')
                    #1x1 > 7x1 > 1x7 > 7x1 > 1x7
                    branch_2 = _conv(x,filters_out=out_filters[4],ksize=1,bn_cfg=self.opts,padding='same',name='branch_2_0')
                    branch_2 = _conv(branch_2,filters_out=out_filters[5],ksize=[7,1],bn_cfg=self.opts,padding='same',name='branch_2_1')
                    branch_2 = _conv(branch_2,filters_out=out_filters[6],ksize=[1,7],bn_cfg=self.opts,padding='same',name='branch_2_2')  
                    branch_2 = _conv(branch_2,filters_out=out_filters[7],ksize=[7,1],bn_cfg=self.opts,padding='same',name='branch_2_3')
                    branch_2 = _conv(branch_2,filters_out=out_filters[8],ksize=[1,7],bn_cfg=self.opts,padding='same',name='branch_2_4')                    
                    #pool > 1x1
                    branch_3 = _pool(x, ksize=3,stride=1,model='avg',padding='SAME',name='branch_3_0')
                    branch_3 = _conv(branch_3,filters_out=out_filters[9],ksize=1,bn_cfg=self.opts,padding='same',name='branch_3_1')   
                    return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
     
     def _block_v3_3(self,x,out_filters,name=None):
          with tf.variable_scope(name):
               if 'module1' in name:
                    #1x1 > 3x3
                    branch_0 = _conv(x,filters_out=out_filters[0],ksize=1,bn_cfg=self.opts,padding='same',name='branch_0_0')
                    branch_0 = _conv(branch_0,filters_out=out_filters[1],ksize=1,stride=2,bn_cfg=self.opts,padding='same',name='branch_0_1')
                    #1x1 > 1x7 > 7x1 > 3x3
                    branch_1 = _conv(x,filters_out=out_filters[2],ksize=1,bn_cfg=self.opts,padding='same',name='branch_1_0')
                    branch_1 = _conv(branch_1,filters_out=out_filters[3],ksize=[1,7],bn_cfg=self.opts,padding='same',name='branch_1_1')
                    branch_1 = _conv(branch_1,filters_out=out_filters[4],ksize=[7,1],bn_cfg=self.opts,padding='same',name='branch_1_2') 
                    branch_1 = _conv(branch_1,filters_out=out_filters[5],ksize=3,stride=2,bn_cfg=self.opts,padding='same',name='branch_1_3')   
                    #pool
                    branch_2 = _pool(x, ksize=3,stride=2,model='max',padding='SAME',name='branch_2')     
                    return tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
               else:
                    #1x1
                    branch_0 = _conv(x,filters_out=out_filters[0],ksize=1,bn_cfg=self.opts,padding='same',name='branch_0')
                    #1x1 > 1x3 > 3x1
                    branch_1_0 = _conv(x,filters_out=out_filters[1],ksize=1,bn_cfg=self.opts,padding='same',name='branch_1_0')
                    branch_1_1 = _conv(branch_1_0,filters_out=out_filters[2],ksize=[1,3],bn_cfg=self.opts,padding='same',name='branch_1_1')
                    branch_1_2 = _conv(branch_1_0,filters_out=out_filters[3],ksize=[3,1],bn_cfg=self.opts,padding='same',name='branch_1_2')
                    branch_1 = tf.concat(axis=3, values=[branch_1_1, branch_1_2])
                    #1x1 > 3x3 > 1x3 > 3x1
                    branch_2_0 = _conv(x,filters_out=out_filters[4],ksize=1,bn_cfg=self.opts,padding='same',name='branch_2_0')
                    branch_2_1 = _conv(branch_2_0,filters_out=out_filters[5],ksize=3,bn_cfg=self.opts,padding='same',name='branch_2_1')
                    branch_2_2 = _conv(branch_2_1,filters_out=out_filters[6],ksize=[1,3],bn_cfg=self.opts,padding='same',name='branch_2_2')  
                    branch_2_3 = _conv(branch_2_1,filters_out=out_filters[7],ksize=[3,1],bn_cfg=self.opts,padding='same',name='branch_2_3')
                    branch_2 = tf.concat(axis=3, values=[branch_2_2, branch_2_3])                  
                    #pool > 1x1
                    branch_3 = _pool(x, ksize=3,stride=1,model='avg',padding='SAME',name='branch_3_0')
                    branch_3 = _conv(branch_3,filters_out=out_filters[8],ksize=1,bn_cfg=self.opts,padding='same',name='branch_3_1')   
                    return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
          
     def _build_inception_v1(self):
          all_fn_list = []
          all_fn_list.append(partial(self._init_block,out_filters=64,name='init_block'))
          for i,key in enumerate(cfg[self.model]):
               all_fn_list.append(partial(self._block_v1,out_filters=cfg[self.model][key],name='block_{}'.format(i+1)))
               if key in ['3b','4e']:
                    all_fn_list.append(partial(_pool,ksize=3,stride=2,model='max',name='pool_{}'.format(i+1)))
          all_fn_list.append(partial(self._final_block,ksize=7,name='final_block'))
          return all_fn_list 
     
     def _build_inception_v2(self):
          return NotImplemented
     
     def _build_inception_v3(self):
          block_func = {'block1':self._block_v3_1,
                        'block2':self._block_v3_2,
                        'block3':self._block_v3_3}          
          all_fn_list = []
          all_fn_list.append(partial(self._init_block,out_filters=32,name='init_block'))     
          for i,block in enumerate(cfg[self.model]):
               for j,key in enumerate(cfg[self.model][block]):
                    all_fn_list.append(partial(block_func[block],out_filters=cfg[self.model][block][key],name='block{}/module{}'.format(i+1,j+1)))
                    # all_fn_list.append(partial(_conv,filters_out=cfg[self.model][block][key][-1],ksize=1,bn_cfg=self.opts,padding='same',name='mconv{}_{}'.format(i+1,j+1)))
          all_fn_list.append(partial(self._final_block,ksize=7,name='final_block'))
          return all_fn_list 
     
     def _build_inception_v4(self):
          return NotImplemented     
     
     def _build_function_list(self):
          func = {'V1':self._build_inception_v1,
                  'V2':self._build_inception_v2,
                  'V3':self._build_inception_v3,
                  'V4':self._build_inception_v4}
          return func.get(self.opts['model_version'])()     
     
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
     parser.add_argument("--model-version", type=str, default="V1",
                        help="The InceptionNet version(V1,V2,V3,V4)")         
     return parser

def set_defaults(opts):
     opts['summary_str'] += "Inception\n"
 
     if not opts.get("base_learning_rate"):
          opts["base_learning_rate"] = -6  
     if not opts.get("learning_rate_schedule"):
          opts["learning_rate_schedule"] = [0.5, 0.75]
     if not opts.get("learning_rate_decay"):
          opts["learning_rate_decay"] = [1.0, 0.1, 0.01]  
               
     if not opts.get('epochs') and not opts.get('iterations'):
          opts['epochs'] = 100        
     
     if not opts.get("BN_decay"):
          opts["BN_decay"] = 0.97     
     if not (
         opts.get("group_norm") is True or opts.get("batch_norm") is True
     ):
         # set group norm as default for ImageNet
          opts["group_norm"] = False
          opts["batch_norm"] = True
     if opts.get("group_norm"):
          if not opts.get("groups"):
               opts["groups"] = 32     
     
     if not opts.get("batch_size"):
          opts['batch_size'] = 4
          
     if not opts.get("model_version"):
          opts['model_version'] = 'V1'     
          
     if (opts['precision'] == '32.32') and not opts.get("shards"):
          opts['shards'] = 2

     opts['name'] = "SN_bs{}".format(opts['batch_size'])

     if not (opts["batch_norm"] or opts["group_norm"]):
          opts["name"] += "_noBN"
          opts["summary_str"] += " No Batch Norm\n"
     elif opts["group_norm"]:
          opts["name"] += "_GN{}".format(opts["groups"])
          opts["summary_str"] += " Group Norm\n" "  {groups} groups\n"
     else:
          opts["name"] += "_BN"
          opts["summary_str"] += " Batch Norm\n"
          if (
             opts["BN_decay"] and opts["BN_decay"] != 0.97
            ):  # defined and not default
               opts["summary_str"] += "  Decay: {}\n".format(opts["BN_decay"])
                    

     if opts.get('replicas') > 1:
          opts['name'] += "x{}r".format(opts['replicas'])
     if opts['pipeline_depth'] > 1:
          opts['name'] += "x{}p".format(opts['pipeline_depth'])
     elif opts.get('gradients_to_accumulate') > 1:
          opts['name'] += "x{}a".format(opts['gradients_to_accumulate'])

     opts['name'] += '_{}{}'.format(opts['precision'],
                                   '_noSR' if opts['no_stochastic_rounding'] else '')
               
