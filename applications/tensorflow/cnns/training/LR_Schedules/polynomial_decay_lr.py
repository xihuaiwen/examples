# Copyright 2019 Graphcore Ltd.
import tensorflow as tf
import math

class LearningRate:
    def __init__(self, opts, total_iterations):
        self.initial_lr = opts["poly_lr_initial_lr"]
        self.decay_steps = opts["poly_lr_decay_steps"]
        self.power = opts["poly_lr_decay_power"]
        self.end_learning_rate = opts["poly_lr_end_lr"]

        self.warmup_iterations = 0
        if opts['warmup_epochs'] > 0:
            if opts['epochs']:
                self.warmup_iterations = total_iterations * opts["warmup_epochs"] // opts["epochs"]
            else:
                opts['warmup_epochs'] = 0

    def feed_dict_lr(self, iteration):
        self.decay_steps = self.decay_steps * math.ceil((iteration+1) / self.decay_steps)
        lr = (self.initial_lr - self.end_learning_rate) * math.pow((1 - iteration / self.decay_steps) , (self.power)) + self.end_learning_rate
        if iteration < self.warmup_iterations:
            return (iteration * lr) / self.warmup_iterations
        else:
            return lr

    def tf_learning_rate_schedule(self, lr):
        global_step_var = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                          trainable=False, initializer=tf.zeros_initializer())

        increment_global_step_op = tf.assign(
            global_step_var, global_step_var + 1)
        initial_lr = self.initial_lr
        end_learning_rate = self.end_learning_rate
        power = self.power
        decay_steps = self.decay_steps

        with tf.control_dependencies([increment_global_step_op]):
            return tf.train.polynomial_decay(
                learning_rate=initial_lr,
                global_step=global_step_var,
                decay_steps=decay_steps,
                end_learning_rate=end_learning_rate,
                power=power
            )


def add_arguments(parser):
    lr_group = parser.add_argument_group(
        'Polynomial Decay Learning Rate. Use with --lr-schedule poly_decay_tf.')

    lr_group.add_argument('--poly-lr-decay-steps', type=int,
                          help="Number of steps in which to reach final learning rate.")
    lr_group.add_argument('--poly-lr-decay-power', type=float,
                          help="Exponent of polynomial decribing the decay. Default 1.0 (linear decay).")
    lr_group.add_argument('--poly-lr-initial-lr', type=float,
                          help="Initial learning rate, before decay.")
    lr_group.add_argument('--poly-lr-end-lr', type=float,
                          help="Final learning rate, after poly-lr-decay-steps.")
    lr_group.add_argument('--warmup-epochs', type=int, default=5,
                          help="Warmup length in epochs (Default=5, set to 0 for no warmup)")
    return parser


def set_defaults(opts):
        # We only need to set defaults for the following if the user has specified a polynomial learning rate
    if not opts["poly_lr_initial_lr"]:
        opts["poly_lr_initial_lr"] = 0.01
    if not opts['poly_lr_decay_steps']:
        opts['poly_lr_decay_steps'] = 100000
    if not opts["poly_lr_end_lr"]:
        opts["poly_lr_end_lr"] = 0.0001
    if not opts["poly_lr_decay_power"]:
        opts["poly_lr_decay_power"] = 1

    opts['summary_str'] += "Polynomial decay applied to learning rate with exponent {poly_lr_decay_power}. Initial rate of {poly_lr_initial_lr}, decaying to {poly_lr_end_lr} after {poly_lr_decay_steps} steps.\n"

    return opts
