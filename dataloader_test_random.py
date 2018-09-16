"""
The dataloader file for sample image from standard MSCOCO test set randomly
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random

import torch
import torch.utils.data as data

import multiprocessing

def get_npy_data(ix, fc_file, att_file, use_att):
    if use_att == True:
        return (np.load(fc_file), np.load(att_file)['feat'], ix)
    else:
        return (np.load(fc_file), np.zeros((1,1,1)), ix)

class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, False) #val2014,test2014
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word


    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size

        self.use_att = getattr(opt, 'use_att', True)
        self.seq_per_img = 1
        # load the json file which contains additional information about the dataset
        #print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        #print('vocab size is ', self.vocab_size) 

        self.input_fc_dir = os.path.join('data','fc')
        self.input_att_dir = os.path.join('data','att')

        # separate out indexes for each of the provided splits
        self.split_ix = {'test2014': []}
        #print(len(self.info['val2014']))
        self.split_name = opt.split
        print(self.split_name)

        for ix in range(len(self.info)):
            img = self.info[ix]
            self.split_ix['test2014'].append(ix)

        print('assigned %d images to split test' %len(self.split_ix['test2014']))

        self.iterators = {'test2014': 0}
        
        self._prefetch_process = {} # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, False)
            # Terminate the child process when the parent exists
        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]
        import atexit
        atexit.register(cleanup)

    def get_batch(self, split, batch_size=None):
        batch_size = batch_size or self.batch_size

        fc_batch = [] # np.ndarray((batch_size * seq_per_img, self.opt.fc_feat_size), dtype = 'float32')
        att_batch = [] # np.ndarray((batch_size * seq_per_img, 14, 14, self.opt.att_feat_size), dtype = 'float32')

        wrapped = False

        infos = []
        gts = []

        for i in range(batch_size):
            # fetch image
            tmp_fc, tmp_att, ix, tmp_wrapped = self._prefetch_process[split].get()
            fc_batch += [tmp_fc] 
            att_batch += [tmp_att] 

            if tmp_wrapped:
                wrapped = True

            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info[ix]['image_id']

            info_dict['filename'] = self.info[ix]['filename']
            infos.append(info_dict)
            #print(i, time.time() - t_start)

        data = {}
        #print(np.shape(np.stack(fc_batch)))
        data['fc_feats'] = np.stack(fc_batch)
        data['att_feats'] = np.stack(att_batch)

        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max':  len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        return data

    # It's not coherent to make DataLoader a subclass of Dataset, but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according the index.
    # However, it's minimum change to switch to pytorch data loading.
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index #self.split_ix[index]
        #print(self.info['images'][ix]['id'])
        return get_npy_data(ix, \
                os.path.join(self.input_fc_dir, str(self.info[ix]['image_id']) + '.npy'),
                os.path.join(self.input_att_dir, str(self.info[ix]['image_id']) + '.npz'),
                self.use_att
                )

    def __len__(self):
        return len(self.info['images'])

class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 0, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                            batch_size=1,
                                            sampler=self.dataloader.split_ix[self.split][self.dataloader.iterators[self.split]:],
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=multiprocessing.cpu_count(),
                                            collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped
    
    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()

        assert tmp[2] == ix, "ix not equal"

        return tmp + [wrapped]
