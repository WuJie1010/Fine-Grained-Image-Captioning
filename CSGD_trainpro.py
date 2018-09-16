"""
Training Captioning Model by Integrating Content Sensitive and Global Discriminative Objective
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import time
import os
from six.moves import cPickle
import pickle
import argparse

import RL_opts
from coco_vocab import Vocabulary
from vse_model import VSE
import models
from knn_dataloader import *
import evalpro_utils
import misc.utils as utils

from misc.rewards_HDR import get_hd_reward # Holistic Discriminative Reward
from misc.rewards_BDR import get_bd_reward # Batch Discriminative Reward
from misc.rewards_CSR import get_cs_reward # Content Sensitive Reward

opt = RL_opts.parse_opt()

# Load Vocabulary Wrapper
vocab = pickle.load(open(os.path.join('./data/mscoco_vocab.pkl'), 'rb'))
opt.vocab_size = len(vocab)

def train(opt):

    """
    :param   caption decoder
    :param   VSE model : image encoder + caption encoder
    """

    """   loading VSE model    """
    # Construct the model
    vse = VSE(opt)
    opt.best = os.path.join('./vse/model_best.pth.tar')
    print("=> loading best checkpoint '{}'".format(opt.best))
    checkpoint = torch.load(opt.best)
    vse.load_state_dict(checkpoint['model'])
    vse.val_start()

    """   loading caption model    """
    opt.use_att = utils.if_use_att(opt.caption_model)

    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    save_path = os.path.join(opt.checkpoint_path,'CSGD')

    if not os.path.exists(save_path):
        os.makedirs(save_path, 0777)

    infos = {}
    histories = {}

    RL_trainmodel = os.path.join('RL_%s' % opt.caption_model)
    
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        start_from_path = os.path.join(opt.start_from,'CSGD')
        with open(os.path.join(start_from_path,'infos_'+opt.id+'.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(start_from_path, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(start_from_path, 'histories_'+opt.id+'.pkl')) as f:
                histories = cPickle.load(f)

    with open(os.path.join(RL_trainmodel,'MLE','infos_'+opt.id+'-best.pkl')) as f:
        infos_XE = cPickle.load(f)
        opt.learning_rate = infos_XE['opt'].current_lr
    
    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    print(loader.iterators)

    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    model = models.setup_pro(opt)

    if vars(opt).get('start_from', None) is not None:

        start_from_path = os.path.join(opt.start_from,'CSGD')
        # check if all necessary files exist 
        assert os.path.isdir(opt.start_from)," %s must be a path" % opt.start_from
        assert os.path.isfile(os.path.join(start_from_path,"infos_"+opt.id+".pkl")),"infos.pkl file does not exist in path %s"%opt.start_from
        assert os.path.isfile(os.path.join(start_from_path,"optimizer.pth")) ,"optimizer.pth.file does not exist in path %s"%opt.start_from

        model_path = os.path.join(start_from_path,'model.pth')
        optimizer_path = os.path.join(start_from_path,'optimizer.pth')

    else:
        model_path = os.path.join(RL_trainmodel,'MLE', 'model-best.pth')
        optimizer_path = os.path.join(RL_trainmodel,'MLE','optimizer-best.pth')

    model.load_state_dict(torch.load(model_path))
    print("model load from {}".format(model_path))  
  
    model.cuda()
    update_lr_flag = True
    # Assure in training mode
    model.train()

    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()
  
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    optimizer.load_state_dict(torch.load(optimizer_path))
    print("optimizer load from {}".format(optimizer_path))   

    all_cider = 0 # for computing the average CIDEr score
    all_dis = 0 # for computing the discriminability percentage

    while True:
        if update_lr_flag:
                # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
                utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
            else:
                opt.current_lr = opt.learning_rate
            # Assign the scheduled sampling prob
            model.ss_prob = 0.25
            print('learning_rate: %s' %str(opt.current_lr))
            update_lr_flag = False

            # start self critical training
            sc_flag = True

        data = loader.get_batch('train')

        torch.cuda.synchronize()
        start = time.time()

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['knn_fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['knn_att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]

        tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
        fc_feats, att_feats, knn_fc_feats, knn_att_feats = tmp

        optimizer.zero_grad()

        gen_result, sample_logprobs = model.sample_score(fc_feats, att_feats, loader, {'sample_max': 0})
        gen_result_baseline, sample_b_logprobs = model.sample_score(fc_feats, att_feats, loader, {'sample_max': 0})
        bd_reward, sample_loss = get_bd_reward(vse, model, fc_feats, att_feats, data, gen_result,gen_result_baseline, loader)
        hd_reward = get_hd_reward(vse, model, fc_feats, knn_fc_feats, data, gen_result,gen_result_baseline, loader)

        cs_reward, m_cider = get_cs_reward(model, fc_feats, att_feats, data, gen_result, gen_result_baseline, loader)
        reward = cs_reward - opt.hdr_w * hd_reward - opt.bdr_w * bd_reward
        loss = rl_crit(sample_logprobs, gen_result, Variable(torch.from_numpy(reward).float().cuda(), requires_grad=False))
        dis_number = (sample_loss < 0.4).float()
        dis_number = dis_number.data.cpu().numpy().sum()
        all_dis += dis_number
        all_cider += m_cider

        loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        train_loss = loss.data[0]

        torch.cuda.synchronize()
        end = time.time()
        print("iter {} (epoch {}), hdr = {:.3f}, bdr = {:.3f}, csr = {:.3f}, time/batch = {:.3f}" \
            .format(iteration, epoch, np.mean(hd_reward[:,0]), np.mean(bd_reward[:,0]), np.mean(cs_reward[:,0]), end - start))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):

            loss_history[iteration] = np.mean(reward[:,0])
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            eval_kwargs = {'split': 'val', 'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = evalpro_utils.eval_split(model, crit, loader, eval_kwargs)
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False
            if True: # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True

                save_path1 = os.path.join(save_path, 'model.pth')
                if not os.path.exists(os.path.dirname(save_path1)):
                    os.makedirs(os.path.dirname(save_path1))
                torch.save(model.state_dict(), save_path1)
                print("model saved to {}".format(save_path1))

                optimizer_path1 = os.path.join(save_path, 'optimizer.pth')
                if not os.path.exists(os.path.dirname(optimizer_path1)):
                    os.makedirs(os.path.dirname(optimizer_path1))
                torch.save(optimizer.state_dict(), optimizer_path1)
                print("optimizer saved to {}".format(optimizer_path1))

                all_dis = all_dis / opt.save_checkpoint_every
                print("all_dis:%f" %all_dis)
                infos['all_dis'] = all_dis

                all_cider = all_cider / opt.save_checkpoint_every
                print("all_cider:%f" %all_cider)
                infos['all_cider'] = all_cider

                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(save_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(save_path, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    save_path2 = os.path.join(save_path, 'model-best.pth')
                    torch.save(model.state_dict(), save_path2)
                    optimizer_path2 = os.path.join(save_path, 'optimizer-best.pth')
                    torch.save(optimizer.state_dict(), optimizer_path2)
                    print("model saved to {}".format(save_path2))
                    with open(os.path.join(save_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)
                    with open(os.path.join(save_path,'histories_'+opt.id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(histories, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

train(opt)
