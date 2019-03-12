
# coding: utf-8

# In[1]:

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
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch


# In[2]:

class Opt():
    def __init__(self):
        self.model = './log_st/model-best_attention_wehao.pth'
        self.cnn_model = 'resnet152' 
        self.infos_path = './log_st/infos__attention_wehao.pkl'
        self.batch_size = 20 
        self.num_images = -1 
        self.language_eval = 0  
        self.dump_images = 1 
        self.dump_json =1
        self.dump_path = 0
        
        
        self.sample_max = 1 #
        
        
        self.beam_size = 2
        self.temperature = 1.0
        self.image_folder = '/home/jiangqy/EXP/competition/data/captions/ai_challenger_caption_test1_20170923/caption_test1_images_20170923'
        self.image_root  = ''
        self.input_fc_dir = ''
        self.input_att_dir  = ''

        self.input_label_h5 = ''
        self.input_json = ''
        self.split = 'test'

        self.coco_json = ''
        self.id = '' 
        self.pretrained_weight= 1    ##### xud

# In[3]:

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
opt = Opt() 
opt.infos_path = './log_st/infos_-best_attention_wehao.pkl'
# Load infos
with open(opt.infos_path) as f:
    infos = cPickle.load(f) 


# In[4]:

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
########  xud
opt.pretrained_weight = infos['opt'].pretrained_weight
ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval", "pretrained_weight"]
#######################

# In[5]:

for k in vars(infos['opt']).keys():
    if k not in ignore:
        if k in vars(opt):
            assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
        else:
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping


# In[6]:

from dataloaderraw import *
if len(opt.image_folder) == 0:
    loader = DataLoader(opt)
else:
    loader = DataLoaderRaw({'folder_path': opt.image_folder, 
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})

# In[7]:

# model


# In[8]:

loader.ix_to_word = infos['vocab']


# In[9]:

# Setup the model
model = models.setup(opt)
# opt.batch_size = 20
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()
crit = utils.LanguageModelCriterion() 


# In[10]:

opt.caption_model


# In[11]:

opt.beam_size = 6

loss, split_predictions, lang_stats = eval_utils.eval_split_test(model, crit, loader, vars(opt))


# In[12]:

for item in split_predictions:
    item['caption'] = (item['caption']).replace(' ','')
    
json.dump(split_predictions, open('./res/res_attention_wenhao_dense_28.json', 'w'))


# In[13]:

opt.model


# In[14]:


'''
    def get_batch(self, split, batch_size=None):
batch_size = batch_size or self.batch_size

# pick an index of the datapoint to load next
fc_batch = np.ndarray((batch_size, 2048), dtype = 'float32')
att_batch = np.ndarray((batch_size, 14, 14, 2048 ), dtype = 'float32')   #####JieZhang
max_index = self.N 
wrapped = False
infos = []
		#a = h5py.File('/home/jiangqy/PycharmProjects/Multi_Label_Model_Python3/data/test_image_ids_hdf5.h5', "r")
		#b =  h5py.File('/home/jiangqy/PycharmProjects/Multi_Label_Model_Python3/data/test_hdf5.h5', "r")
for i in range(batch_size):
    ri = self.iterator
    ri_next = ri + 1
    if ri_next >= max_index:
        ri_next = 0
        wrapped = True
        # wrap back around
    self.iterator = ri_next
    
    
    
    
    filepath = self.files[ri].split('/')[-1]
    
   # print(self.files[ri],filepath) 
    
    img = skimage.io.imread(self.files[ri])

    if len(img.shape) == 2:
        img = img[:,:,np.newaxis]
        img = np.concatenate((img, img, img), axis=2)

    img = img.astype('float32')/255.0
    img = torch.from_numpy(img.transpose([2,0,1])).cuda()
    img = Variable(preprocess(img), volatile=True)
    tmp_fc, tmp_att = self.my_resnet(img)
    
    if filepath == self.a[u'test image_id'][ri]:
        fc_batch[i] = self.b['test fc'][ri]
    else:
        for k in range(30000):
            if filepath == self.a[u'test image_id'][k]:
                fc_batch[i] = self.b['test fc'][k]
                break
    #fc_batch[i] = tmp_fc.data.cpu().float().numpy()
    att_batch[i] = tmp_att.data.cpu().float().numpy()

    info_struct = {}
    info_struct['id'] = self.ids[ri]
    info_struct['file_path'] = self.files[ri]
    infos.append(info_struct)

data = {}
data['fc_feats'] = fc_batch
data['att_feats'] = att_batch
data['bounds'] = {'it_pos_now': self.iterator, 'it_max': self.N, 'wrapped': wrapped}
data['infos'] = infos 
return data 
'''        

