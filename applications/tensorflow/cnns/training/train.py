# Copyright 2019 Graphcore Ltd.
"""
Training CNNs on Graphcore IPUs.

See the README and the --help option for more information.
"""
import tensorflow as tf
import os
import re
import time
import argparse
import datetime
import random
from socket import gethostname
from collections import deque, OrderedDict, namedtuple
from functools import partial
import numpy as np
import sys
import importlib
import validation
import log as logging
from tensorflow.python import ipu
from ipu_utils import get_config
from tensorflow.python.ipu.autoshard import automatic_sharding
from tensorflow.python.ipu import loops, ipu_infeed_queue, ipu_outfeed_queue, ipu_compiler
from tensorflow.python.ipu.ipu_optimizer import CrossReplicaOptimizer
from tensorflow.python.ipu.gradient_accumulation_optimizer import GradientAccumulationOptimizer
from tensorflow.python.ipu.utils import reset_ipu_seed
from tensorflow.python.ipu.ops import pipelining_ops
from ipu_optimizer import IPUOptimizer
from tensorflow.python.ipu.scopes import ipu_scope
import Datasets.data as dataset
DATASET_CONSTANTS = dataset.DATASET_CONSTANTS

#tf_report
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from gc_profile import save_tf_report
with tf.device('cpu'):
    report = gen_ipu_ops.ipu_event_trace()

#true to use virtul ipu
IPU_MODEL = False
if IPU_MODEL:
    os.environ['TF_POPLAR_FLAGS'] = "--use_ipu_model"

GraphOps = namedtuple(
    'graphOps', ['graph',
                 'session',
                 'init',
                 'ops',
                 'placeholders',
                 'iterator',
                 'outfeed',
                 'saver'])

pipeline_schedule_options = [str(p).split(".")[-1] for p in list(pipelining_ops.PipelineSchedule)]


def calculate_loss(logits, label, opts):
    predictions = tf.argmax(logits, 1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, label), tf.float16))

    # Loss
    if opts["label_smoothing"] > 0:
        num_classes = logits.get_shape().as_list()[1]
        smooth_negatives = opts["label_smoothing"] / (num_classes - 1)
        smooth_positives = (1.0 - opts["label_smoothing"])
        smoothed_labels = tf.one_hot(label, num_classes, on_value=smooth_positives, off_value=smooth_negatives)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=smoothed_labels))
    else:
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label))

    tf.add_to_collection('losses', cross_entropy)
    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return loss, cross_entropy, accuracy

def calculate_multi_loss(logits, label, opts):
    predictions = tf.argmax(logits[0], 1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, label), tf.float16))
    cross_entropy = 0
    # Loss
    for each_logics in logits:
        if opts["label_smoothing"] > 0:
            num_classes = each_logics.get_shape().as_list()[1]
            smooth_negatives = opts["label_smoothing"] / (num_classes - 1)
            smooth_positives = (1.0 - opts["label_smoothing"])
            smoothed_labels = tf.one_hot(label, num_classes, on_value=smooth_positives, off_value=smooth_negatives)
            cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=each_logics, labels=smoothed_labels))
        else:
            cross_entropy += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=each_logics, labels=label))

    tf.add_to_collection('losses', cross_entropy)
    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return loss, cross_entropy, accuracy


def get_optimizer(opts):
    if opts['optimiser'] == 'SGD':
        opt_fun = tf.train.GradientDescentOptimizer
    elif opts['optimiser'] == 'momentum':
        opt_fun = partial(tf.train.MomentumOptimizer, momentum=opts['momentum'])

    wd_exclude = opts["wd_exclude"] if "wd_exclude" in opts.keys() else []

    def filter_fn(name):
        return not any(s in name for s in wd_exclude)

    return lambda lr: IPUOptimizer(opt_fun(lr),
                                   sharded=opts["shards"] > 1 and opts['pipeline_depth'] == 1,
                                   replicas=opts["replicas"],
                                   gradients_to_accumulate=opts["gradients_to_accumulate"] * opts['pipeline_depth'],
                                   pipelining = opts['pipeline_depth'] > 1,
                                   weight_decay=opts["weight_decay"] * opts['loss_scaling'],
                                   weight_decay_filter_fn=filter_fn)


def calculate_and_apply_gradients(loss, opts=None, learning_rate=None):
    optimizer = get_optimizer(opts)(learning_rate / opts['loss_scaling'])
    grads_and_vars = optimizer.compute_gradients(loss * opts['loss_scaling'])
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        return learning_rate, optimizer.apply_gradients(grads_and_vars)


def basic_training_step(image, label, model, opts, learning_rate):
    """
    A basic training step that will work on all hardware
    """
    logits = model(opts, training=True, image=image)
    if type(logits)==list: 
        loss, cross_entropy, accuracy = calculate_multi_loss(logits, label, opts)
    else:
        loss, cross_entropy, accuracy = calculate_loss(logits, label, opts)

    learning_rate, train_op = calculate_and_apply_gradients(loss, opts, learning_rate=learning_rate)
    if opts['shards'] > 1:
        def filter(edge):
            return (any(f in e for e in edge for f in opts["sharding_exclude_filter"]) or
                    not any(f in e for e in edge for f in opts["sharding_include_filter"]))
        automatic_sharding(opts['shards'], image, cross_entropy, edge_filter=filter)

    return loss, cross_entropy, accuracy, learning_rate, train_op


def basic_pipelined_training_step(model, opts, learning_rate, infeed, outfeed, iterations_per_step=1):

    def first_stage(learning_rate, image, label,  pipeline_stage=None):
        return learning_rate, pipeline_stage(image), label,

    def final_stage(learning_rate, x, label,  pipeline_stage=None):
        x = pipeline_stage(x)
        loss, cross_entropy, accuracy = calculate_loss(x, label, opts)
        return loss, cross_entropy, accuracy, learning_rate / opts["loss_scaling"]

    model_stages = model(opts)
    computational_stages = [partial(first_stage, pipeline_stage=model_stages[x]) for x in range(len(model_stages) - 1)]
    computational_stages.append(partial(final_stage, pipeline_stage=model_stages[-1]))

    def optimizer_function(loss, _, __, lr):
        optimizer = get_optimizer(opts)(lr)
        return pipelining_ops.OptimizerFunctionOutput(optimizer, loss * opts["loss_scaling"])

    options = None
    amps = opts['available_memory_proportion']
    if amps and len(amps) > 1:
        # Map values to the different pipeline stages
        options = []
        for i in range(len(amps) // 2):
            options.append(pipelining_ops.PipelineStageOptions({"availableMemoryProportion": amps[2*i]},
                                                               {"availableMemoryProportion": amps[2*i + 1]}))

    return pipelining_ops.pipeline(computational_stages=computational_stages,
                                   pipeline_depth=int(opts['pipeline_depth']),
                                   repeat_count=iterations_per_step,
                                   inputs=[learning_rate],
                                   infeed_queue=infeed,
                                   outfeed_queue=outfeed,
                                   optimizer_function=optimizer_function,
                                   forward_propagation_stages_poplar_options=options,
                                   backward_propagation_stages_poplar_options=options,
                                   pipeline_schedule=next(p for p in list(pipelining_ops.PipelineSchedule)
                                                          if opts["pipeline_schedule"] == str(p).split(".")[-1]),
                                   name="Pipeline")


def training_step_with_infeeds_and_outfeeds(train_iterator, outfeed, model, opts, learning_rate, iterations_per_step=1):
    """
    Training step that uses an infeed loop with outfeeds. This runs 'iterations_per_step' steps per session call. This leads to
    significant speed ups on IPU. Not compatible with running on CPU or GPU.
    """

    if int(opts['pipeline_depth']) > 1:
        training_step = partial(basic_pipelined_training_step,
                                model=model.staged_model,
                                opts=opts,
                                learning_rate=learning_rate,
                                infeed=train_iterator,
                                outfeed=outfeed,
                                iterations_per_step=iterations_per_step)


        return ipu_compiler.compile(training_step, [])
    else:
        training_step = partial(basic_training_step,
                                model=model.Model,
                                opts=opts,
                                learning_rate=learning_rate)

        def training_step_loop(image=None, label=None, outfeed=None):
            loss, cross_ent, accuracy, lr_out, apply_grads = training_step(image, label)
            outfeed = outfeed.enqueue((loss, cross_ent, accuracy, lr_out))
            return outfeed, apply_grads

        def compiled_fn():
            return loops.repeat(iterations_per_step,
                                partial(training_step_loop, outfeed=outfeed),
                                [],
                                train_iterator)


        return ipu_compiler.compile(compiled_fn, [])


def training_graph(model, opts, iterations_per_step=1):

    train_graph = tf.Graph()
    with train_graph.as_default():
        placeholders = dict()
        datatype = tf.float16 if opts["precision"].split('.')[0] == '16' else tf.float32
        placeholders['learning_rate'] = tf.placeholder(datatype, shape=[])
        learning_rate = placeholders['learning_rate']

        placeholders['loss_value'] = tf.placeholder(dtype=datatype, name='loss_value') 
        placeholders['auc_value'] = tf.placeholder(dtype=datatype, name='auc_value') 

        # datasets must be defined outside the ipu device scope
        train_iterator = ipu_infeed_queue.IPUInfeedQueue(dataset.data(opts, is_training=True),
                                                         feed_name='training_feed',
                                                         replication_factor=opts['replicas'])
        outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(feed_name="outfeed",
                                                          replication_factor=opts['replicas'])


        with ipu_scope('/device:IPU:0'):
            train = training_step_with_infeeds_and_outfeeds(train_iterator, outfeed_queue, model,
                                                            opts, learning_rate, iterations_per_step)

        outfeed = outfeed_queue.dequeue()

        logging.print_trainable_variables(opts)

        train_saver = tf.train.Saver(max_to_keep=999999)

        ipu.utils.move_variable_initialization_to_cpu()
        train_init = tf.global_variables_initializer()

    globalAMP = None
    if opts["available_memory_proportion"] and len(opts["available_memory_proportion"]) == 1:
        globalAMP = opts["available_memory_proportion"][0]

    ipu_options = get_config(ipu_id=opts["select_ipu"],
                             prng=not opts["no_stochastic_rounding"],
                             shards=opts["shards"],
                             number_of_replicas=opts['replicas'],
                             max_cross_replica_buffer_size=opts["max_cross_replica_buffer_size"],
                             fp_exceptions=opts["fp_exceptions"],
                             xla_recompute=opts["xla_recompute"],
                             seed=opts["seed"],
                             availableMemoryProportion=globalAMP,
                             profiling=opts["profiling"],
                             profile_execution=opts["profile_execution"])

    ipu.utils.configure_ipu_system(ipu_options)
    train_sess = tf.Session(graph=train_graph, config=tf.ConfigProto())
    

    return GraphOps(train_graph, train_sess, train_init, [train], placeholders, train_iterator, outfeed, train_saver)


def training_step(train, e, learning_rate):
    # Run Training
    start = time.time()
    _ = train.session.run(train.ops, feed_dict={train.placeholders['learning_rate']: learning_rate})
    batch_time = (time.time() - start)
    if not os.environ.get('TF_POPLAR_FLAGS') or '--use_synthetic_data' not in os.environ.get('TF_POPLAR_FLAGS'):
        loss, cross_ent, accuracy, lr_out = train.session.run(train.outfeed)
        loss = np.mean(loss)
        accuracy = 100.0 * np.mean(accuracy)
        lr = lr_out.flatten()[-1]
    else:
        loss, accuracy, lr = 0, 0, 0
    return loss, accuracy, batch_time, lr


#定义变量数据汇总函数，函数附带变量的直方图分布
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def train_process(model, LR_Class, opts):

    # --------------- OPTIONS ---------------------
    epochs = opts["epochs"]
    iterations_per_epoch = DATASET_CONSTANTS[opts['dataset']]['NUM_IMAGES'] // opts["total_batch_size"]
    if not opts['iterations']:
        iterations = epochs * iterations_per_epoch
        log_freq = iterations_per_epoch // opts['logs_per_epoch']
    else:
        iterations = opts['iterations']
        log_freq = opts['log_freq']

    if log_freq < opts['batches_per_step']:
        iterations_per_step = log_freq
    else:
        iterations_per_step = log_freq // int(round(log_freq / opts['batches_per_step']))

    iterations_per_valid = iterations_per_epoch
    iterations_per_ckpt = iterations_per_epoch // opts['ckpts_per_epoch'] if opts['ckpts_per_epoch'] else np.inf

    LR = LR_Class(opts, iterations)

    batch_accs = deque(maxlen=iterations_per_epoch // iterations_per_step)
    batch_losses = deque(maxlen=iterations_per_epoch // iterations_per_step)
    batch_times = deque(maxlen=iterations_per_epoch // iterations_per_step)
    start_all = None

    # -------------- BUILD TRAINING GRAPH ----------------

    train = training_graph(model, opts, iterations_per_step*opts["gradients_to_accumulate"])
    train.session.run(train.init)
    train.session.run(train.iterator.initializer)


    # 生成一个写日志的writer，并将当前的TensorFlow计算图写入日志。TensorFlow提供了
    train_loss_summary = tf.summary.scalar(name='train_loss', tensor=train.placeholders['loss_value'])    
    train_auc_summary = tf.summary.scalar(name='train_auc', tensor=train.placeholders['auc_value'])    
    writer = tf.summary.FileWriter('./logs/', train.session.graph)
   
    # -------------- BUILD VALIDATION GRAPH ----------------

    if opts['validation']:
        valid = validation.initialise_validation(model, opts)

    # -------------- SAVE AND RESTORE --------------


    if opts['ckpts_per_epoch']:
        filepath = train.saver.save(train.session, opts["checkpoint_path"], global_step=0)
        print("Saved checkpoint to {}".format(filepath))

    if opts.get('restoring'):
        filename_pattern = re.compile(".*ckpt-[0-9]+$")
        ckpt_pattern = re.compile(".*ckpt-([0-9]+)$")
        filenames = sorted([os.path.join(opts['logs_path'], f[:-len(".index")])
                            for f in os.listdir(opts['logs_path']) if filename_pattern.match(f[:-len(".index")]) and f[-len(".index"):] == ".index"], key=lambda x: int(ckpt_pattern.match(x).groups()[0]))
        latest_checkpoint = filenames[-1]
        logging.print_to_file_and_screen("Restoring training from latest checkpoint: {}".format(latest_checkpoint), opts)
        ckpt_pattern = re.compile(".*ckpt-([0-9]+)$")
        i = int(ckpt_pattern.match(latest_checkpoint).groups()[0]) + 1
        train.saver.restore(train.session, latest_checkpoint)
        epoch = float(opts["total_batch_size"] * (i + iterations_per_step)) / DATASET_CONSTANTS[opts['dataset']][
            'NUM_IMAGES']
    else:
        i = 0

    # ------------- TRAINING LOOP ----------------

    print_format = ("step: {step:6d}, iteration: {iteration:6d}, epoch: {epoch:6.2f}, lr: {lr:6.4g}, loss: {loss_avg:6.3f}, top-1 accuracy: {train_acc_avg:6.3f}%"
                    ", img/sec: {img_per_sec:6.2f}, time: {it_time:8.6f}, total_time: {total_time:8.1f}")

    step = 0
    start_all = time.time()
    while i < iterations:
        step += opts["gradients_to_accumulate"]
        log_this_step = ((i // log_freq) < ((i + iterations_per_step) // log_freq) or
                         (i == 0) or
                         ((i + (2 * iterations_per_step)) >= iterations))
        ckpt_this_step = ((i // iterations_per_ckpt) < ((i + iterations_per_step) // iterations_per_ckpt) or
                          (i == 0) or
                          ((i + (2 * iterations_per_step)) >= iterations))
        valid_this_step = (opts['validation'] and
                           ((i // iterations_per_valid) < ((i + iterations_per_step) // iterations_per_valid) or
                           (i == 0) or
                           ((i + (2 * iterations_per_step)) >= iterations)))

        # Run Training
        try:
            batch_loss, batch_acc, batch_time, current_lr = training_step(train, i + 1, LR.feed_dict_lr(i))

            out = train.session.run(report)
            save_tf_report(out)

            if opts['pipeline_depth'] > 1:
                current_lr *= opts["loss_scaling"]
        except tf.errors.OpError as e:
            raise tf.errors.ResourceExhaustedError(e.node_def, e.op, e.message)

        batch_time /= iterations_per_step

        # Calculate Stats
        batch_accs.append([batch_acc])
        batch_losses.append([batch_loss])

        if i != 0:
            batch_times.append([batch_time])

        # Print loss
        if log_this_step:
            train_acc = np.mean(batch_accs)
            train_loss = np.mean(batch_losses)

            train_loss_summary_value, train_auc_summary_value = train.session.run([train_loss_summary, train_auc_summary], feed_dict={train.placeholders['loss_value']:train_loss, train.placeholders['auc_value']:train_acc}) #
            writer.add_summary(train_auc_summary_value, global_step=i)  #注意：一次只能add一个summary
            writer.add_summary(train_loss_summary_value, global_step=i)

            if len(batch_times) != 0:
                avg_batch_time = np.mean(batch_times)
            else:
                avg_batch_time = batch_time

            # flush times every time it is reported
            batch_times.clear()

            total_time = time.time() - start_all
            epoch = float(opts["total_batch_size"] * (i + iterations_per_step)) / DATASET_CONSTANTS[opts['dataset']]['NUM_IMAGES']

            stats = OrderedDict([
                ('step', step),
                ('iteration', i+iterations_per_step),
                ('epoch', epoch),
                ('lr', current_lr),
                ('loss_batch', batch_loss),
                ('loss_avg', train_loss),
                ('train_acc_batch', batch_acc),
                ('train_acc_avg', train_acc),
                ('it_time', avg_batch_time),
                ('img_per_sec', opts['total_batch_size']/avg_batch_time),
                ('total_time', total_time),
            ])

            logging.print_to_file_and_screen(print_format.format(**stats), opts)
            logging.write_to_csv(stats, i == 0, True, opts)

        if ckpt_this_step:
            filepath = train.saver.save(train.session, opts["checkpoint_path"], global_step=i+iterations_per_step)
            print("Saved checkpoint to {}".format(filepath))

        # Eval
        if valid_this_step and opts['validation']:
            if 'validation_points' not in locals():
                validation_points = []
            validation_points.append((i + iterations_per_step, epoch, i == 0, filepath))

        i += iterations_per_step

    # ------------ RUN VALIDATION ------------
    if 'validation_points' in locals() and opts['validation']:
        for iteration, epoch, first_run, filepath in validation_points:
            validation.validation_run(valid, filepath, iteration, epoch, first_run, opts)

    # --------------- CLEANUP ----------------
    train.session.close()


def add_main_arguments(parser, required=True):
    group = parser.add_argument_group('Main')
    group.add_argument('--model', default='resnet', help="Choose model")
    group.add_argument('--lr-schedule', default='stepped',
                       help="Learning rate schedule function. Default: stepped")
    group.add_argument('--help', action='store_true', help='Show help information')

    return parser


def set_main_defaults(opts):
    opts['summary_str'] = "\n"


def add_training_arguments(parser):
    tr_group = parser.add_argument_group('Training')
    tr_group.add_argument('--batch-size', type=int,
                          help="Set batch-size for training graph")
    tr_group.add_argument('--gradients-to-accumulate', type=int, default=1,
                          help="Number of gradients to accumulate before doing a weight update")
    tr_group.add_argument('--base-learning-rate', type=float,
                          help="Base learning rate exponent (2**N). blr = lr /  bs")
    tr_group.add_argument('--epochs', type=float,
                          help="Number of training epochs")
    tr_group.add_argument('--iterations', type=int, default=None,
                          help="Force a fixed number of training iterations to be run rather than epochs.")
    tr_group.add_argument('--weight-decay', type=float, default=1e-4,
                          help="Value for weight decay bias, setting to 0 removes weight decay.")
    tr_group.add_argument('--loss-scaling', type=float, default=128,
                          help="Loss scaling factor")
    tr_group.add_argument('--label-smoothing', type=float, default=0,
                          help="Label smoothing factor (Default=0 => no smoothing)")

    tr_group.add_argument('--ckpts-per-epoch', type=int, default=1,
                          help="Checkpoints per epoch")
    tr_group.add_argument('--no-validation', action="store_false", dest='validation',
                          help="Dont do any validation runs.")
    tr_group.set_defaults(validation=True)
    tr_group.add_argument('--shards', type=int, default=1,
                          help="Number of IPU shards for training graph")
    tr_group.add_argument('--replicas', type=int, default=1,
                          help="Replicate graph over N workers to increase batch to batch-size*N")
    tr_group.add_argument('--max-cross-replica-buffer-size', type=int, default=10*1024*1024,
                          help="""The maximum number of bytes that can be waiting before a cross
                                replica sum op is scheduled. [Default=10*1024*1024]""")

    tr_group.add_argument('--pipeline-depth', type=int, default=1,
                          help="Depth of pipeline to use. Must also set --shards > 1.")
    tr_group.add_argument('--pipeline-splits', nargs='+', type=str, default=None,
                          help="Strings for splitting pipelines. E.g. b2/0/relu b3/0/relu")
    tr_group.add_argument('--pipeline-schedule', type=str, default="Interleaved",
                          choices=pipeline_schedule_options, help="Pipelining scheduler. Choose between 'Interleaved' and 'Grouped'")
    tr_group.add_argument('--optimiser', type=str, default="SGD", choices=['SGD', 'momentum'],
                          help="Optimiser")
    tr_group.add_argument('--momentum', type=float, default=0.9,
                          help="Momentum coefficient")
    return parser


def set_training_defaults(opts):
    if int(opts['pipeline_depth']) > 1:
        opts['gradients_to_accumulate'] = 1

    opts['total_batch_size'] = opts['batch_size']*opts['gradients_to_accumulate']*opts['pipeline_depth']*opts['replicas']
    opts['summary_str'] += "Training\n"
    opts['summary_str'] += " Batch Size: {total_batch_size}\n"
    if opts['pipeline_depth'] > 1:
        opts['summary_str'] += "  Pipelined over {shards} stages with depth {pipeline_depth} \n"
    elif opts['gradients_to_accumulate'] > 1:
        opts['summary_str'] += "  Accumulated over {gradients_to_accumulate} fwds/bwds passes \n"
    if opts['replicas'] > 1:
        opts['summary_str'] += "  Training on {replicas} workers \n"
    opts['summary_str'] += (" Base Learning Rate: 2**{base_learning_rate}\n"
                            " Weight Decay: {weight_decay}\n"
                            " Loss Scaling: {loss_scaling}\n")
    if opts["iterations"]:
        opts['summary_str'] += " Iterations: {iterations}\n"
    else:
        opts['summary_str'] += " Epochs: {epochs}\n"

    if opts['shards'] > 1:
        opts['summary_str'] += " Training Shards: {shards}\n"

    if opts['optimiser'] == 'SGD':
        opts['summary_str'] += "SGD\n"
    elif opts['optimiser'] == 'momentum':
        opts['name'] += '_Mom'
        opts['summary_str'] += ("SGD with Momentum\n"
                                " Momentum Coefficient: {momentum}\n")


def add_ipu_arguments(parser):
    group = parser.add_argument_group('IPU')
    group.add_argument('--precision', type=str, default="16.16", choices=["16.16", "16.32", "32.32"],
                       help="Precision of Ops(weights/activations/gradients) and Master data types: 16.16, 16.32, 32.32")
    group.add_argument('--no-stochastic-rounding', action="store_true",
                       help="Disable Stochastic Rounding")
    group.add_argument('--batches-per-step', type=int, default=1000,
                       help="Maximum number of batches to perform on the device before returning to the host.")
    group.add_argument('--select-ipu', type=str, default="AUTO",
                       help="Select IPU either: AUTO or IPU ID")
    group.add_argument('--sharding-exclude-filter', nargs='+', type=str, default=['0/p'],
                       help="Excluded strings for splitting edges (only a substring must match)")
    group.add_argument('--sharding-include-filter', nargs='+', type=str, default=[''],
                       help="Included strings for splitting edges (only a substring must match)")
    group.add_argument('--fp-exceptions', action="store_true",
                       help="Turn on floating point exceptions")
    group.add_argument('--xla-recompute', action="store_true",
                       help="Allow recomputation of activations to reduce memory usage")
    group.add_argument('--seed', default=None, help="Seed for randomizing training")
    group.add_argument('--available-memory-proportion', default=None, nargs='+',
                       help="Proportion of memory which is available for convolutions. Use a value of less than 0.6 "
                            "to reduce memory usage.")
    group.add_argument('--profiling', action="store_true",
                       help="Turn on profiling exceptions")
    group.add_argument('--profile-execution', action="store_true", help="Allow execution for profine")
    return parser


def set_ipu_defaults(opts):
    opts['summary_str'] += "Using Infeeds\n Max Batches Per Step: {batches_per_step}\n"
    opts['summary_str'] += 'Device\n'
    opts['summary_str'] += ' Precision: {}{}\n'.format(opts['precision'], '_noSR' if opts['no_stochastic_rounding'] else '')
    opts['summary_str'] += ' IPU\n'
    opts['poplar_version'] = os.popen('popc --version').read()
    opts['summary_str'] += ' {poplar_version}'
    if opts['select_ipu'] == 'AUTO':
        opts['select_ipu'] = -1

    opts["profiling"] = True
    opts["profile_execution"] = True

    opts['hostname'] = gethostname()
    opts['datetime'] = str(datetime.datetime.now())

    if opts['seed']:
        # Seed the various random sources
        seed = int(opts['seed'])
        opts['seed_specified'] = opts['seed'] is not None
        random.seed(seed)
        # Set other seeds to different values for extra safety
        tf.set_random_seed(random.randint(0, 2**32 - 1))
        np.random.seed(random.randint(0, 2**32 - 1))
        reset_ipu_seed(random.randint(-2**16, 2**16 - 1))
        opts['seed'] = seed
    else:
        opts['seed_specified'] = False

    opts['summary_str'] += (' {hostname}\n'
                            ' {datetime}\n')


def create_parser(model, lr_schedule, parser):
    parser = model.add_arguments(parser)
    parser = dataset.add_arguments(parser)
    parser = add_training_arguments(parser)
    parser = lr_schedule.add_arguments(parser)
    parser = add_ipu_arguments(parser)
    parser = logging.add_arguments(parser)
    return parser


def set_defaults(model, LR, opts):
    set_main_defaults(opts)
    dataset.set_defaults(opts)
    model.set_defaults(opts)
    set_training_defaults(opts)
    LR.set_defaults(opts)
    validation.set_validation_defaults(opts)
    set_ipu_defaults(opts)
    logging.set_defaults(opts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN Training in TensorFlow', add_help=False)
    parser = add_main_arguments(parser)
    args, unknown = parser.parse_known_args()
    args = vars(args)

    try:
        model = importlib.import_module("Models." + args['model'])
    except ImportError:
        raise ValueError('Models/{}.py not found'.format(args['model']))

    try:
        lr_schedule = importlib.import_module("LR_Schedules." + args['lr_schedule'])
    except ImportError:
        raise ValueError("LR_Schedules/{}.py not found".format(args['lr_schedule']))

    # Large number of deprecation warnings that cannot be resolved yet.
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = create_parser(model, lr_schedule, parser)
    opts = vars(parser.parse_args())
    if opts['help']:
        parser.print_help()
    else:
        if opts['gradients_to_accumulate'] > 1 and opts['pipeline_depth'] > 1:
            raise ValueError("gradients-to-accumulate can't be specified when using --pipeline-depth > 1")
        if opts['pipeline_depth'] > 1 and opts['shards'] == 1:
            raise ValueError("--pipeline-depth can only be used if --shards > 1")
        amps = opts['available_memory_proportion']
        if amps and len(amps) > 1:
            if not opts['pipeline_depth'] > 1:
                raise ValueError('--available-memory-propotion should only have one value unless using pipelining')
            if len(amps) != int(opts['shards']) * 2:
                raise ValueError('--available-memory-propotion should have either one value or 2*shards values specified')

        opts["command"] = ' '.join(sys.argv)
        set_defaults(model, lr_schedule, opts)

        logging.print_to_file_and_screen("Command line: " + opts["command"], opts)
        logging.print_to_file_and_screen(opts["summary_str"].format(**opts), opts)
        opts["summary_str"] = ""
        logging.print_to_file_and_screen(opts, opts)
        train_process(model, lr_schedule.LearningRate, opts)
