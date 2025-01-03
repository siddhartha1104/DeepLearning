#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Logging."""

import builtins
import decimal
import logging
import os
import simplejson
import sys

from pycls.config import cfg

import pycls.utils.distributed as du
import pycls.utils.metrics as mu
import pdb

# Show filename and line number in logs
_FORMAT = '[%(filename)s: %(lineno)3d]: %(message)s'

# Log file name (for cfg.LOG_DEST = 'file')
_LOG_FILE = 'stdout.log'

# Printed json stats lines will be tagged w/ this
_TAG = 'json_stats: '


def _suppress_print():
    """Suppresses printing from the current process."""

    def ignore(*_objects, _sep=' ', _end='\n', _file=sys.stdout, _flush=False):
        pass

    builtins.print = ignore


def setup_logging():
    """Sets up the logging."""
    # Enable logging only for the master process
    if du.is_master_proc():
        # Clear the root logger to prevent any existing logging config
        # (e.g. set by another module) from messing with our setup
        logging.root.handlers = []
        # Construct logging configuration
        logging_config = {
            'level': logging.INFO,
            'format': _FORMAT
        }
        # Log either to stdout or to a file
        if cfg.LOG_DEST == 'stdout':
            logging_config['stream'] = sys.stdout
        else:
            logging_config['filename'] = os.path.join(cfg.OUT_DIR, _LOG_FILE)
        # Configure logging
        logging.basicConfig(**logging_config)
    else:
        pass
        # _suppress_print()


def get_logger(name):
    """Retrieves the logger."""
    return logging.getLogger(name)


def log_json_stats(stats, cur_epoch=None, writer=None, is_epoch=False, params=0, flops=0, model=None, is_master=False):
    """Logs json stats."""
    if writer is not None:
        for k, v in stats.items():
            if isinstance(v, float) or isinstance(v, int):
                writer.add_scalar(k, v, cur_epoch + 1)
        # if model is not None:
        #     for name, param in model.named_parameters():
        #         writer.add_histogram(name, param.clone().cpu().data.numpy(), cur_epoch)
    # Decimal + string workaround for having fixed len float vals in logs
    stats = {
        k: decimal.Decimal('{:.6f}'.format(v)) if isinstance(v, float) else v
        for k, v in stats.items()
    }
    json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
    logger = get_logger(__name__)
    logger.info('{:s}{:s}'.format(_TAG, json_stats))
    if is_epoch and cur_epoch is not None and is_master:
        epoch_id = cur_epoch + 1
        result_info = ', '.join(
            [str(round(params / 1000000, 3)), str(round(flops / 1000000000, 3)), '{:.3f}'.format(stats['time_avg']),
             '{:.3f}'.format(stats['top1_err']), '{:.3f}'.format(stats['top5_err']),
             str(cfg.RGRAPH.GROUP_NUM), str(cfg.RGRAPH.DIM_LIST[0]), str(cfg.RGRAPH.SEED_TRAIN)])
        with open("{}/results_epoch{}.txt".format(cfg.OUT_DIR, epoch_id), "a") as text_file:
            text_file.write(result_info + '\n')


def load_json_stats(log_file):
    """Loads json_stats from a single log file."""
    with open(log_file, 'r') as f:
        lines = f.readlines()
    json_lines = [l[l.find(_TAG) + len(_TAG):] for l in lines if _TAG in l]
    json_stats = [simplejson.loads(l) for l in json_lines]
    return json_stats


def parse_json_stats(log, row_type, key):
    """Extract values corresponding to row_type/key out of log."""
    vals = [row[key] for row in log if row['_type'] == row_type and key in row]
    if key == 'iter' or key == 'epoch':
        vals = [int(val.split('/')[0]) for val in vals]
    return vals


def get_log_files(log_dir, name_filter=''):
    """Get all log files in directory containing subdirs of trained models."""
    names = [n for n in sorted(os.listdir(log_dir)) if name_filter in n]
    files = [os.path.join(log_dir, n, _LOG_FILE) for n in names]
    f_n_ps = [(f, n) for (f, n) in zip(files, names) if os.path.exists(f)]
    files, names = zip(*f_n_ps)
    return files, names
