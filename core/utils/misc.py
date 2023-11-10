import argparse
from collections import OrderedDict
import errno
import os
import numpy as np
from PIL import Image

import torch
from core.configs import cfg


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def parse_args():
    parser = argparse.ArgumentParser(description="Unsupervised Particle Simulator Domain Adaptation")
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()

    if args.opts is not None and args.opts != []:
        args.opts[-1] = args.opts[-1].strip('\r\n')

    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return args




def load_checkpoint(model, path, module='feature_extractor'):
    print("Loading checkpoint from {}".format(path))
    if str(path).endswith('.ckpt'):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))['state_dict']
        model_weights = {k: v for k, v in checkpoint.items() if k.startswith(module)}
        model_weights = OrderedDict([[k.split(module + '.')[-1], v.cpu()] for k, v in model_weights.items()])
        model.load_state_dict(model_weights)
    elif str(path).endswith('.pth'):
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        model_weights = checkpoint[module]
        model_weights = strip_prefix_if_present(checkpoint[module], 'module.')
        model.load_state_dict(model_weights)
    else:
        raise NotImplementedError('Only support .ckpt and .pth file')


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict
