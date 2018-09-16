from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['Show_tell', 'all_img', 'fc']:
        return False
    return True

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out

def decode_att_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = []
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                txt.append(ix_to_word[str(ix)])
            else:
                break
        out.append(txt)
    return out

# Input: seq, D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_1dsequence(ix_to_word, seq):
    D = len(seq)
    txt = []
    for j in range(D):
        ix = seq[j]
        if ix > 0 :
            #if j >= 1:
                #txt = txt + ' '
            txt.append(ix_to_word[str(ix)])
        else:
            break
    return txt

def sequence(seq):
    D = len(seq)
    txt = []
    for j in range(D):
        ix = seq[j]
        if ix > 0 :
            #if j >= 1:
                #txt = txt + ' '
            txt.append(str(ix))
        else:
            break
    return txt

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1) 
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * Variable(mask)
        output = torch.sum(output) / torch.sum(mask)

        return output

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]  #80*17
        mask =  mask[:, :input.size(1)]    #80*17
        input = to_contiguous(input).view(-1, input.size(2))  #80*17,9488
        target = to_contiguous(target).view(-1, 1)   #80*17,1
        mask = to_contiguous(mask).view(-1, 1)   #80*17,1
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        #print(group['params'])
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)
