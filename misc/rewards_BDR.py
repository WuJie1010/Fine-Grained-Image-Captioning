"""
Batch Discriminative Reward
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

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()


def get_bd_reward(vse, model, fc_feats, att_feats, data, gen_result,gen_result_baseline, loader):

    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img  128
    zero = torch.LongTensor([0] * batch_size)

    # process the sample results
    gen_result = torch.cat([gen_result.cpu(), zero],1)  # add 0 to end the sentence
    lengths_gen = np.zeros(batch_size, dtype=np.int)
    for i, cap in enumerate(gen_result.numpy()):
        for j in range(len(cap)):
            if cap[j] == 0:
                lengths_gen[i] = j+1 # summary the sentence lengths
                break

    # sort the data according to length number
    data = []
    for i in range(batch_size):
        data.append((fc_feats[i], gen_result[i], lengths_gen[i], i)) # add i to readjust the order
    data.sort(key=lambda x: x[2], reverse=True) # sort according to sentence length
    img_feats, sample_result, lengths_gen, index = zip(*data) # get the reshape data

    img_feats = torch.stack(img_feats, 0)  # stack the image feature
    sample_result = torch.stack(sample_result, 0) # stack the sample results

    # compute the embeddings
    img_emb, cap_emb = vse.forward_emb(img_feats, sample_result, lengths_gen) #embbing
    loss = vse.forward_loss(img_emb, cap_emb) # get the loss as the reward

    # resort the data according to length number, for corresponding loss
    redata = []
    for i in range(batch_size):
        redata.append((img_feats[i], loss[i], index[i]))
    redata.sort(key=lambda x: x[2], reverse=False) # reshape according the index in line 41, to get the raw inout order
    feats, sample_loss, reindex = zip(*redata) # get the loss for corresponding order
    sample_loss = torch.stack(sample_loss, 0) # stack the loss
    # process another sample results
    # get sample baseline

    baseline_res = torch.cat([gen_result_baseline.cpu(), zero],1)
    lengths_gre = np.zeros(batch_size, dtype=np.int)
    for i, cap in enumerate(baseline_res.numpy()):
        for j in range(len(cap)):
            if cap[j] == 0:
                lengths_gre[i] = j+1
                break

    # sort the data according to length number
    gre_data = []
    for i in range(batch_size):
        gre_data.append((fc_feats[i], baseline_res[i], lengths_gre[i], i))
    gre_data.sort(key=lambda x: x[2], reverse=True)
    img_feats, baseline_res, lengths_gre, index = zip(*gre_data)

    img_feats = torch.stack(img_feats, 0)
    baseline_res = torch.stack(baseline_res, 0)

    # compute the embeddings
    img_emb, cap_emb = vse.forward_emb(img_feats, baseline_res, lengths_gre)
    baseline_loss = vse.forward_loss(img_emb, cap_emb)

    # resort the data according to length number
    gre_redata = []
    for i in range(batch_size):
        gre_redata.append((img_feats[i], baseline_loss[i], index[i]))
    gre_redata.sort(key=lambda x: x[2], reverse=False)
    feats, greedy_loss, reindex = zip(*gre_redata)
    greedy_loss = torch.stack(greedy_loss, 0)

    distinct_loss = sample_loss - greedy_loss
    distinct_loss = distinct_loss.data.cpu().numpy()
    print('Batch Distinct scores:', np.mean(distinct_loss))

    rewards = np.repeat(distinct_loss, gen_result.shape[1]-1, 1)
    #print(np.shape(rewards))
    return rewards, sample_loss
