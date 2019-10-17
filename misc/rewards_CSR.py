"""
Local Discriminative Reward
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import misc.utils as utils
from collections import OrderedDict
import torch
from torch.autograd import Variable

import sys
sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD_token import CiderD

CiderD_scorer = CiderD(df='coco-train-idxs')
#CiderD_scorer = CiderD(df='corpus')

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def get_cs_reward(model, fc_feats, att_feats, data, gen_result,gen_result_baseline,loader):
 
    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img  80
    seq_per_img = batch_size // len(data['gts'])

    res = OrderedDict()

    #sents = utils.decode_ngrm_sequence(loader.get_vocab(), gen_result)
    #sents_1 = utils.decode_sequence(loader.get_vocab(), gen_result_baseline)

    #gen_result = gen_result.cpu().numpy()
    greedy_res = gen_result_baseline.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]
    gts = OrderedDict()

    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}

    scores, log_scores, token_scores = CiderD_scorer.compute_score(gts, res, loader)

    m_cider = np.mean(scores[:batch_size])
    print('Cider scores:', m_cider)
    print('logCider scores:', np.mean(log_scores[:batch_size]))

    token_rewards = np.zeros((2 * batch_size, gen_result.shape[1]))

    for i in range(len(token_scores)):
        for j in range(len(token_scores[i])):
            if j<gen_result.shape[1]:
                token_rewards[i][j] = token_scores[i][j]
    rewards = token_rewards[:batch_size] - token_rewards[batch_size:]

    log_scores = log_scores[:batch_size] - log_scores[batch_size:]
    rewards_score = np.repeat(log_scores[:, np.newaxis], gen_result.shape[1], 1)

    all_reward = rewards + rewards_score
    #print(all_reward)
    return all_reward, m_cider
