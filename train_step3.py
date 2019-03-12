#! -- coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import argparse
import time
import os
from six.moves import cPickle
import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils
import jieba

try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None

def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'



class Opt():
    def __init__(self):
        self.input_json = 'data/coco.json'
        self.input_fc_h5 = 'data/coco_ai_challenger_talk_fc_.h5'

        self.input_att_h5 = 'data/coco_ai_challenger_talk_att_.h5'
        self.input_label_h5 = 'data/coco_ai_challenger_talk_label_.h5'

        self.start_from = None
        self.caption_model = 'topdown'
        self.rnn_size = 512
        self.num_layers =  1
        self.rnn_type = 'lstm'

        self.input_encoding_size = 512
        self.att_hid_size = 512
        self.fc_feat_size = 2048

        self.att_feat_size = 2048

        self.max_epochs = 10
        self.batch_size = 5
        self.grad_clip = 0.1

        self.drop_prob_lm = 0.5
        self.drop_prob_lm_input = 0.3
        self.seq_per_img  = 5

        self.beam_size = 5
        self.optim = 'adam'
        self.learning_rate  = 1e-5
        self.learning_rate_decay_start  = -1
        self.learning_rate_decay_every = 3
        self.learning_rate_decay_rate = 0.8
        self.optim_alpha = 0.9
        self.optim_beta = 0.999
        self.optim_epsilon = 1e-8
        self.weight_decay = 0
        self.id = 'st'
        self.scheduled_sampling_start  = -1
        self.scheduled_sampling_increase_every = 5
        self.scheduled_sampling_max_prob  = 0.25

        self.val_images_use = 3200
        self.save_checkpoint_every = 2500
        self.checkpoint_path = 'save'

        self.language_eval = 0
        self.losses_log_every = 25
        self.load_best_score = 1

        self.id = ''
        self.train_only = 0
        self.pretrained_weight = None




opt = Opt()
opt.caption_model = 'topdown' #'topdown_residual' or 'topdown'
opt.input_json = '/data/xud/json_preprocess_data/coco_ai_challenger_talk.json'
opt.input_fc_h5 = '/data/xud/json_preprocess_data/coco_ai_challenger_talk_resnet151_fc_.h5'
opt.input_att_h5 = '/data/xud/json_preprocess_data/coco_ai_challenger_talk_resnet151_att_.h5'
opt.input_label_h5 = '/data/xud/json_preprocess_data/coco_ai_challenger_talk_resnet151_label_.h5'

opt.batch_size  = 20
opt.learning_rate  = 4e-4 # 4e-4 # 8e-5 for restore
opt.learning_rate_decay_every = 3
opt.learning_rate_decay_start = 0

opt.scheduled_sampling_start  = 200
opt.checkpoint_path  = './log_st/setting15/'
#opt.start_from = './log_st/setting13/'
if not os.path.exists(opt.checkpoint_path):
    os.makedirs(opt.checkpoint_path)
opt.save_checkpoint_every =  5000
opt.val_images_use = 1000
opt.max_epochs = 40
opt.use_att  = utils.if_use_att(opt.caption_model)




loader = DataLoader(opt)
opt.vocab_size = loader.vocab_size
print("vocab size: ", opt.vocab_size)
opt.seq_length = loader.seq_length
tf_summary_writer = tf and tf.summary.FileWriter(opt.checkpoint_path)

# -------------------  xud ---------------------------
ix_to_word = loader.ix_to_word
word_to_ix = {}
for k, v in ix_to_word.items():
    word_to_ix[v.encode("utf-8")] = int(k.encode("utf-8"))

pretrained_weight = 0.08 * np.random.random((opt.vocab_size+1, opt.input_encoding_size)) - 0.04
f = open("data/imageCaption_512d_iter100.embed", "r")
_ = f.readline()
lines = f.readlines()
print("embedding: ",len(lines))
check = {}
for line in lines:
    sp = line.split()
    word = sp[0]
    embed = np.zeros(opt.input_encoding_size)
    for i in range(opt.input_encoding_size):
        embed[i] = float(sp[i+1])
    if word_to_ix.get(word) is not None:
        pretrained_weight[word_to_ix[word], :] = embed
        check[word_to_ix[word]] = True
f.close()
opt.pretrained_weight = pretrained_weight
print("vocab check: ", len(check))
#-----------------------------------------------------



infos = {}
histories = {}
if opt.start_from is not None:
    # open old infos and check if models are compatible
    with open(os.path.join(opt.start_from, 'infos_'+opt.id+'_attention.pkl')) as f:
        infos = cPickle.load(f)
        saved_model_opt = infos['opt']
        need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
        for checkme in need_be_same:
            assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

    if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
        with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')) as f:
            histories = cPickle.load(f)


val_result_history = histories.get('val_result_history', {})
loss_history = histories.get('loss_history', {})
lr_history = histories.get('lr_history', {})
ss_prob_history = histories.get('ss_prob_history', {})

loader.iterators = infos.get('iterators', loader.iterators)
loader.split_ix = infos.get('split_ix', loader.split_ix)
if opt.load_best_score == 1:
    best_val_score = infos.get('best_val_score', None)



model = models.setup(opt)
model = model.cuda() #should be "model = model.cuda()"
update_lr_flag = True
# Assure in training mode
model.train()

crit = utils.LanguageModelCriterion()
optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)



if vars(opt).get('start_from', None) is not None:
    optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

train_loss = []
train_loss_every_validation = []
train_loss_in_one_validation = []
res = []
iter_print_count =200


epoch = 0
iteration = 0
start = time.time()
prev_lr = opt.learning_rate
while True:
    if update_lr_flag:
            # Assign the learning rate
        if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:

            frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
            decay_factor = opt.learning_rate_decay_rate  ** frac
            opt.current_lr = opt.learning_rate * decay_factor
            utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
            if opt.current_lr != prev_lr:
                print("current learning rate:", opt.current_lr)
                prev_lr = opt.current_lr
        else:
            opt.current_lr = opt.learning_rate
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob
            update_lr_flag = False

    # start = time.time()
    # Load data from train split (0)
    data = loader.get_batch('train')
#     print('Read data:', time.time() - start)
    torch.cuda.synchronize()

    tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
    tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
    fc_feats, att_feats, labels, masks = tmp


    optimizer.zero_grad()
#     print('calculate prediction')
    predictions = model(fc_feats,att_feats,labels)
    loss = crit(model(fc_feats, att_feats, labels), labels[:,1:], masks[:,1:])

    loss.backward()

    utils.clip_gradient(optimizer, opt.grad_clip)
    optimizer.step()
    train_loss.append(loss.data[0])
    torch.cuda.synchronize()

    train_loss_in_one_validation.append(loss.data[0])

    if iteration % iter_print_count == 0:
        end = time.time()
        print("iter {} (epoch {}), train_loss = {:.3f}, time = {:.3f}" \
                .format(iteration, epoch, sum(train_loss)/len(train_loss), end - start))
        train_loss = []
        start = time.time()
            # Update the iteration and epoch
    iteration += 1
    if data['bounds']['wrapped']:
        epoch += 1
        update_lr_flag = True
    # Write the training loss summary
    if (iteration % opt.losses_log_every == 0):

        if tf is not None:
            add_summary_value(tf_summary_writer, 'train_loss', loss.data[0], iteration)
            add_summary_value(tf_summary_writer, 'learning_rate', opt.current_lr, iteration)
            add_summary_value(tf_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
            tf_summary_writer.flush()

        loss_history[iteration] = loss.data[0]
        lr_history[iteration] = opt.current_lr
        ss_prob_history[iteration] = model.ss_prob

    # 在验证集上检验效果
    if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            st = time.time()
            eval_kwargs = {'split': 'val',
                            'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(model, crit, loader, eval_kwargs)

                # Write validation result into summary
            # if tf is not None:
            #     add_summary_value(tf_summary_writer, 'validation loss', val_loss, iteration)
            #     for k,v in lang_stats.items():
            #         add_summary_value(tf_summary_writer, k, v, iteration)
            #     tf_summary_writer.flush()
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
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)
                print ("Current Time:  "+str(time.strftime("%d %b %Y %H:%M:%S", time.localtime())))

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
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'_attention.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'_attention.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best_attention.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'best_attention.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)
            print("evaluate validation set cost: "+"{:.4f}".format(time.time() - st)+" seconds")
            train_loss_every_validation.append(sum(train_loss_in_one_validation)/len(train_loss_in_one_validation))
            train_loss_in_one_validation = []
            print("Train loss every validation:")
            print(train_loss_every_validation)
    # Stop if reaching max epochs
    if epoch >= opt.max_epochs and opt.max_epochs != -1:
        break

