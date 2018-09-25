"""
evaluation file
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import models
from knn_dataloader import *
import evalpro_utils
import argparse
import misc.utils as utils
import torch
import linecache

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--caption_model', type=str, default="TDA",
                    help='ST, TDA')
parser.add_argument('--model', type=str, default='',
                help='path to model to evaluate')
parser.add_argument('--CNN', type=str,  default='Resnet101',
                help='Resnet101, Resnet152')
parser.add_argument('--cnn_model', type=str,  default='Resnet101',
                help='Resnet101, Resnet152')
parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate')
parser.add_argument('--datasets', type=str,  default='mscoco',
                    help='mscoco, flickr30k')  
# Basic options
parser.add_argument('--batch_size', type=int, default=0,
                help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=5000,
                help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=1,
                help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=0,
                help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=1,
                help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=1,
                help='Write image paths along with predictions into vis json? (1=yes,0=no)')

parser.add_argument('--self_critical_metric', type=str, default='CIDEr',
                help='RL metric ,CIDEr or BLEU4  BLCIDEr')
# Sampling options
parser.add_argument('--sample_max', type=int, default=1,
                help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--beam_size', type=int, default=3,
                help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--temperature', type=float, default=1.0,
                help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
parser.add_argument('--diversity_lambda', type=float, default=0.5,
                help='used for diverse beam search. Usually from 0.2 to 0.8. Higher value of lambda produces a more diverse list')
parser.add_argument('--max_ppl', type=int, default=0,
                help='beam search by max perplexity or max probability.')
parser.add_argument('--group_size', type=int, default=1,
                help='used for diverse beam search. if group_size is 1, then it\'s normal beam search')
parser.add_argument('--decoding_constraint', type=int, default=1,
                help='If 1, not allowing same word in a row')


# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default='', 
                help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='', 
                help='In case the image paths have to be preprended with a root path to an image folder')
# For evaluation on MSCOCO images from some split:
parser.add_argument('--input_fc_dir', type=str, default='',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_att_dir', type=str, default='',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_label_h5', type=str, default='',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_json', type=str, default='', 
                help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
parser.add_argument('--split', type=str, default='test', 
                help='if running on MSCOCO images, which split to use: val|test|train')
parser.add_argument('--coco_json', type=str, default='', 
                help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
parser.add_argument('--checkpoint_path', type=str, default='save',
                    help='directory to store checkpointed models')# misc
parser.add_argument('--id', type=str, default='', 
                help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')

opt = parser.parse_args()

path = os.path.join(opt.checkpoint_path,'CSGD') #checkpoint_path = RL_
print(path)


Resultpath = os.path.join(path,'Result','Result_eval.txt')
if not os.path.exists(os.path.dirname(Resultpath)):
    os.makedirs(os.path.dirname(Resultpath))

Sentencespath = os.path.join(path,'Result','Sentence_eval.json')

if os.path.exists(os.path.dirname(Sentencespath)):
    predictions = json.load(open(Sentencespath,'r'))
    lang_stats = evalpro_utils.language_eval(predictions)
else:
    os.makedirs(os.path.dirname(Sentencespath))

    opt.infos_path = os.path.join(path,'infos_'+str(opt.caption_model)+'.pkl')
    # Load infos
    with open(opt.infos_path) as f:
        infos = cPickle.load(f)

    opt.save = infos['opt'].save_checkpoint_every

    pth_number = int(infos['iter']/opt.save) #number to decode

    # override and collect parameters
    if len(opt.input_fc_dir) == 0:
        opt.input_fc_dir = os.path.join('data','fc')
        opt.input_att_dir = os.path.join('data','att')
        opt.input_label_h5 = infos['opt'].input_label_h5

    if len(opt.input_json) == 0:
        opt.input_json = infos['opt'].input_json
    if opt.batch_size == 0:
        opt.batch_size = infos['opt'].batch_size
    if len(opt.id) == 0:
        opt.id = infos['opt'].id
    ignore = ["id", "batch_size", "input_fc_dir", "input_att_dir", "beam_size", "start_from", "language_eval"]
    for k in vars(infos['opt']).keys():
        if k not in ignore:
            if k in vars(opt):
                assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
            else:
                vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model


    vocab = infos['vocab'] # ix -> word mapping

    opt.model = os.path.join(path,'model-best.pth')
    model = models.setup_pro(opt)
    model.load_state_dict(torch.load(opt.model))
    model.cuda()
    model.eval()
    crit = utils.LanguageModelCriterion()

    # Create the Data Loader instance
    loader = DataLoader(opt)
    loader.ix_to_word = infos['vocab']

    # Set sample options
    loss, split_predictions, lang_stats = evalpro_utils.eval_split(model, crit, loader, vars(opt))
    if opt.dump_json == 1:
        # dump the json
        json.dump(split_predictions, open(Sentencespath, 'w'))

if lang_stats:
    with open(Resultpath,'w') as fout:
        print(lang_stats, file=fout)


