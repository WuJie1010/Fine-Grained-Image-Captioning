from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

def eval_split(model, data_caption, loader, eval_kwargs={}):
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    # Make sure in the evaluation mode
    model.eval()
    loader.reset_iterator(split)

    n = 0

    all = []
    while True:
        data = loader.get_batch(split)
        #print(data['infos'][0]['ix'])
        n = n + loader.batch_size

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        fc_feats, att_feats = tmp
        # forward the model to also get generated samples for each image

        pro = model.sample_recall(fc_feats, att_feats, loader, data_caption, eval_kwargs)

        if n == loader.batch_size:
            all = pro
        else:
            all = np.concatenate((all,pro))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break
    return all[:5000]
