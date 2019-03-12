from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
# from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



class Opt():
    def __init__(self):
        self.model = './log_st/model-best_attention.pth'
        self.cnn_model = 'resnet152'
        self.infos_path = './log_st/infos_attention.pkl'

        self.batch_size = 20
        self.num_images = -1
        self.language_eval = 0
        self.dump_images = 1
        self.dump_json =1
        self.dump_path = 0


        self.sample_max = 1 #

        self.beam_size = 2
        self.temperature = 1.0
        self.image_folder = '/data/xud/ai_challenger_caption_test1_20170923/caption_test1_images_20170923'
        self.image_root  = ''
        self.input_fc_dir = ''
        self.input_att_dir  = ''

        self.input_label_h5 = ''
        self.input_json = ''
        self.split = 'test'

        self.coco_json = ''
        self.id = ''
        self.pretrained_weight= 1



opt = Opt()
opt.infos_path = './log_st/setting11/infos_best_attention.pkl'
opt.model = './log_st/setting11/model-best_attention.pth'

# Load infos
with open(opt.infos_path) as f:
    infos = cPickle.load(f)



# override and collect parameters
if len(opt.input_fc_dir) == 0:
    opt.input_fc_h5 = infos['opt'].input_fc_h5
    opt.input_att_h5 = infos['opt'].input_att_h5
    opt.input_label_h5 = infos['opt'].input_label_h5
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
if len(opt.id) == 0:
    opt.id = infos['opt'].id

opt.pretrained_weight = infos['opt'].pretrained_weight
ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval", "pretrained_weight"]


for k in vars(infos['opt']).keys():
    if k not in ignore:
        if k in vars(opt):
            assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
        else:
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model
vocab = infos['vocab'] # ix -> word mapping



from dataloaderraw import *
if len(opt.image_folder) == 0:
    loader = DataLoader(opt)
else:
    loader = DataLoaderRaw({'folder_path': opt.image_folder,
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})
loader.ix_to_word = infos['vocab']




# Setup the model
model = models.setup(opt)
# opt.batch_size = 20
print("loading parameters.")
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()
crit = utils.LanguageModelCriterion()



opt.beam_size = 5
loss, split_predictions, lang_stats = eval_utils.eval_split_test(model, crit, loader, vars(opt))

for item in split_predictions:
    item['caption'] = (item['caption']).replace(' ','')

json.dump(split_predictions, open('./res/result_topdown_setting_11_ep20_prob.json', 'w+'))
