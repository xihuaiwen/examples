import numpy as np
import tensorflow as tf
import math

class LearningRate:
    """A cosine learning rate schedule with optional warmup."""
    def __init__(self, opts, total_iterations):
        self.init_lr = opts["learning_rate"]
        self.total_iterations = total_iterations
        self.decay_rate = opts["decay_rate"]
        self.decay_steps = opts["decay_steps"]
        
        self.step = [10000000,30000000,50000000,60000000]

        self.warmup_iterations = 0
        if opts['warmup_epochs'] > 0:
            if opts['epochs']:
                self.warmup_iterations = total_iterations * opts["warmup_epochs"] // opts["epochs"]
            else:
                opts['warmup_epochs'] = 0
                print("--warmup-epochs needs --epochs not --iterations specified. Setting warmup-epochs to zero.")
    def feed_dict_lr(self, iteration):
        #staircase=True-->阶梯型衰减 False-->标准指数型衰减
        lr = self.init_lr*math.pow(self.decay_rate, (iteration/self.decay_steps))
        if iteration < self.warmup_iterations:
            return (iteration * lr) / self.warmup_iterations
        else:
            return lr
        # lr = 0
        # if iteration > self.step[3]:
        #     lr = 0.000001
        # elif iteration > self.step[2]:
        #     lr = 0.00001
        # elif iteration > self.step[1]:
        #     lr = 0.0001
        # else:
        #     lr = 0.001
        # return lr   


def add_arguments(parser):
    lr_group = parser.add_argument_group('Exp Learning Rate')
    lr_group.add_argument('--warmup-epochs', type=int, default=3,
                          help="Warmup length in epochs (Default=5, set to 0 for no warmup)")
    lr_group.add_argument('--decay-steps', type=int, default=5000000,
                          help="Change learning rate until n steps (Default=10)")  
    lr_group.add_argument('--learning-rate', type=float, default=0.01,
                          help="learning rate for exp (Default=0.01)")   
    lr_group.add_argument('--decay-rate', type=float, default=0.9,
                          help="learning rate for exp (Default=0.9)")    
    return parser


def set_defaults(opts):
    opts['summary_str'] += "Exp LR schedule\n"
    if opts["warmup_epochs"] > 0:
        opts['summary_str'] += " Warmup: {} epochs\n".format('{warmup_epochs}')
    else:
        opts['summary_str'] += " No warmup\n"
    if not opts["learning_rate"]:
        opts["learning_rate"] = 0.01
    if not opts["decay_rate"]:
        opts["decay_rate"] = 0.9
    return opts