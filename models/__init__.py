from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import misc.utils as utils
import torch

from .ShowTellModel import ShowTellModel
from .TopDownAttModel import TopDownAttModel

def setup_pro(opt):

    # ShowTellModel
    if opt.caption_model == 'ST':
        model = ShowTellModel(opt)
    # Top-down attention model
    elif opt.caption_model == 'TDA':
        model = TopDownAttModel(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    return model
