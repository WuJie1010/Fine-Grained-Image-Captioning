from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils

def language_eval(preds, model_id, split, checkpoint_path, CNN, datasets):
    import sys
    sys.path.append("coco-caption")
    annFile = 'coco-caption/annotations/captions_val2014.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')
    
    cache_path = 'eval/test.json'
    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))
    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    return out

def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', False)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 1)
    checkpoint_path = eval_kwargs.get('checkpoint_path', 1)
    CNN = eval_kwargs.get('CNN', 1)
    datasets = eval_kwargs.get('datasets', 1)
    # Make sure in the evaluation mode
    model.eval()
    loader.reset_iterator(split)

    n = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    pred_sentences1 = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        fc_feats, att_feats = tmp
        # forward the model to also get generated samples for each image

        seq, _ = model.sample_score(fc_feats, att_feats,loader, eval_kwargs)
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            predictions.append(entry)
        if len(predictions) % 500 == 0:
            print('%d images are decoded' % len(predictions))
        if verbose:
            print('image %s: %s' %(entry['image_id'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()


        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    if lang_eval == 1:
        lang_stats = language_eval(predictions, eval_kwargs['id'], split, checkpoint_path, CNN,datasets)
    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats
