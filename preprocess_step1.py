# encoding: utf-8
from __future__ import print_function
import os
import argparse
import json
from PIL import Image
import jieba


train_caption_json = '/home/jiangqy/EXP/competition/data/captions/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json'
train_img_dir = '/home/jiangqy/EXP/competition/data/captions/ai_challenger_caption_train_20170902/caption_train_images_20170902'
val_caption_json = '/home/jiangqy/EXP/competition/data/captions/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json'
val_img_dir = '/home/jiangqy/EXP/competition/data/captions/ai_challenger_caption_validation_20170910/caption_validation_images_20170910'
test_img_dir = '/home/jiangqy/EXP/competition/data/captions/ai_challenger_caption_test1_20170923/caption_test1_images_20170923'

"""
# change original form into coco form
# save the changed form in out_put_file
def convert2coco(caption_json, img_dir):
    dataset = json.load(open(caption_json, 'r'))
    imgdir = img_dir

    coco = dict()
    coco[u'info'] = { u'desciption':u'AI challenger image caption in mscoco format'}
    coco[u'licenses'] = ['Unknown', 'Unknown']
    coco[u'images'] = list()
    coco[u'annotations'] = list()

    for ind, sample in enumerate(dataset):
        img = Image.open(os.path.join(imgdir, sample['image_id']))
        width, height = img.size

        coco_img = {}
        coco_img[u'license'] = 0 #不用管
        coco_img[u'file_name'] = os.path.split(img_dir)[-1]+'/'+sample['image_id'] #图片的文件名称
        coco_img[u'width'] = width  #图片的宽
        coco_img[u'height'] = height #图片的高
        coco_img[u'date_captured'] = 0
        coco_img[u'coco_url'] = sample['url']  #图片对应的网址，可以输入url找到对应的图片
        coco_img[u'flickr_url'] = sample['url'] #图片对应的网址
        coco_img['id'] = os.path.splitext(os.path.basename(sample['image_id']))[0] #图片的id,我们用图片的名字作为图片的id

        coco_anno = {}
        coco_anno[u'image_id'] = os.path.splitext(os.path.basename(sample['image_id']))[0] #图片的id,我们用图片的名字作为图片的id，这个id用来对应图片与文字
        coco_anno[u'id'] = os.path.splitext(os.path.basename(sample['image_id']))[0] #图片的id,我们用图片的名字作为图片的id，这个id用来对应图片与文字
        coco_anno[u'caption'] = sample['caption'] #图片对应的文字的描述

        coco[u'images'].append(coco_img)
        coco[u'annotations'].append(coco_anno)
        if ind % 1000 == 0:
            print('{}/{}'.format(ind, len(dataset)))

    output_file = os.path.join('/data/xud/json_preprocess_data', 'coco_'+os.path.basename(caption_json))  #存储
    with open(output_file, 'w') as fid:
        json.dump(coco, fid)
    print('Saved to {}'.format(output_file))
    return coco


def convert2coco_val(caption_json, img_dir):
    dataset = json.load(open(caption_json, 'r'))
    imgdir = img_dir

    coco = dict()
    coco[u'info'] = { u'desciption':u'AI challenger image caption in mscoco format'}
    coco[u'licenses'] = ['Unknown', 'Unknown']
    coco[u'images'] = list()
    coco[u'annotations'] = list()

    for ind, sample in enumerate(dataset):
        img = Image.open(os.path.join(imgdir, sample['image_id']))
        width, height = img.size

        coco_img = {}
        coco_img[u'license'] = 0
        coco_img[u'file_name'] = os.path.split(img_dir)[-1]+'/'+sample['image_id']
        coco_img[u'width'] = width
        coco_img[u'height'] = height
        coco_img[u'date_captured'] = 0
        coco_img[u'coco_url'] = sample['url']
        coco_img[u'flickr_url'] = sample['url']
        coco_img['id'] = os.path.splitext(os.path.basename(sample['image_id']))[0]

        coco_anno = {}
        coco_anno[u'image_id'] = os.path.splitext(os.path.basename(sample['image_id']))[0]
        coco_anno[u'id'] = os.path.splitext(os.path.basename(sample['image_id']))[0]
        coco_anno[u'caption'] = sample['caption']
        idx = 0
        for s in sample['caption']:  #该部分很少被运行,主要解决上面那个图中的情况，有些captions中间断开了，为空，那我们就用它上面一句话来补充它形成5句话
            if len(s)==0:
                print('error: some caption had no words?')
                print(coco_img[u'file_name'])
                sample['caption'][idx] = sample['caption'][idx-1]
                print(sample['caption'])
                print(len(sample['caption']),len(coco_anno[u'caption']))
                #break
            idx = idx+1
        coco[u'images'].append(coco_img)
        coco[u'annotations'].append(coco_anno)
        if ind % 1000 == 0:
            print('{}/{}'.format(ind, len(dataset)))

    output_file = os.path.join('/data/xud/json_preprocess_data', 'coco_'+os.path.basename(caption_json))
    with open(output_file, 'w') as fid:
        json.dump(coco, fid)
    print('Saved to {}'.format(output_file))
    return coco



def convert2coco_eval(caption_json, img_dir):
    dataset = json.load(open(caption_json, 'r'))
    imgdir = img_dir

    coco = dict()
    coco[u'info'] = { u'desciption':u'AI challenger image caption in mscoco format'}
    coco[u'licenses'] = ['Unknown', 'Unknown']
    coco[u'images'] = list()
    coco[u'annotations'] = list()
    coco[u'type'] = u'captions'
    for ind, sample in enumerate(dataset):
        #img = Image.open(os.path.join(imgdir, sample['image_id']))
        #width, height = img.size
        width, height = 224, 224

        coco_img = {}
        coco_img[u'license'] = 0
        coco_img[u'file_name'] = os.path.split(img_dir)[-1]+'/'+sample['image_id']
        coco_img[u'width'] = width
        coco_img[u'height'] = height
        coco_img[u'date_captured'] = 0
        coco_img[u'coco_url'] = sample['url']
        coco_img[u'flickr_url'] = sample['url']
        coco_img['id'] = os.path.splitext(os.path.basename(sample['image_id']))[0]

        coco_anno = {}
        coco_anno[u'image_id'] = os.path.splitext(os.path.basename(sample['image_id']))[0]
        coco_anno[u'id'] = os.path.splitext(os.path.basename(sample['image_id']))[0]
        coco_anno[u'caption'] = sample['caption']

        coco[u'images'].append(coco_img)

        for coco_anno_ in coco_anno['caption']:
            coco_anno_s = {}
            coco_anno_s[u'image_id'] = coco_anno[u'image_id']
            coco_anno_s[u'id'] = coco_anno[u'id']
            w = jieba.cut(coco_anno_.strip(), cut_all=False)
            p = ' '.join(w)
            coco_anno_ = p
            coco_anno_s[u'caption'] = coco_anno_
            coco[u'annotations'].append(coco_anno_s)
        if ind % 1000 == 0:
            print('{}/{}'.format(ind, len(dataset)))

    output_file = os.path.join('/data/xud/json_preprocess_data', 'coco_val_'+os.path.basename(caption_json))
    with open(output_file, 'w') as fid:
        json.dump(coco, fid)
    print('Saved to {}'.format(output_file))
    return coco



def convert2coco_test(img_dir):
    coco = dict()
    coco[u'info'] = { u'desciption':u'AI challenger image caption in mscoco format'}
    coco[u'licenses'] = ['Unknown', 'Unknown']
    coco[u'images'] = list()
    ind = 0
    for im_name in enumerate(os.listdir(img_dir)):
        width, height = 224, 224

        coco_img = {}
        coco_img[u'license'] = 0
        coco_img[u'file_name'] = im_name[1]
        coco_img[u'width'] = width
        coco_img[u'height'] = height
        coco_img[u'date_captured'] = 0
        coco_img[u'id'] = os.path.splitext(os.path.basename(im_name[1]))[0]
        ind = ind + 1
        coco[u'images'].append(coco_img)

        print('{}/{}'.format(ind, len(os.listdir(img_dir))))

    output_file = os.path.join('/data/xud/json_preprocess_data', 'ai_challenger_test1.json')
    with open(output_file, 'w') as fid:
        json.dump(coco, fid)
    print('Saved to {}'.format(output_file))
    return coco




convert2coco(train_caption_json, train_img_dir)
convert2coco_val(val_caption_json, val_img_dir)
convert2coco_test(test_img_dir)
convert2coco_eval(val_caption_json, val_img_dir)

"""

import json
import os
import jieba
train = json.load(open('/data/xud/json_preprocess_data/coco_caption_train_annotations_20170902.json', 'r'))
val = json.load(open('/data/xud/json_preprocess_data/coco_caption_validation_annotations_20170910.json', 'r'))
raw_text_file = open("./data/raw_text_to_pretrain_w2v.txt", "w+")
print(val.keys())
print(val['info'])
print(len(val['images']))
print(len(val['annotations']))
print(val['images'][0])
print(val['annotations'][0])

imgs = train['images']+val['images']
annots = train['annotations']+val['annotations']

itoa = {}
for a in annots:
    imgid = a['image_id']
    if not imgid in itoa: itoa[imgid] = []
    itoa[imgid].append(a)

out_json={}
out=[]
for i, img in enumerate(imgs):
    out_im = {}
    # coco specific here, they store train/val images separately
    split = 'train' if 'train' in img['file_name'] else 'val'
    annotsi = itoa[img['id']]
    sentid = 0
    out_im['cocoid'] = img['id']
    out_im['filename'] = os.path.basename(img['file_name'])
    if 'val' in img['file_name']:
        out_im['filepath'] = 'ai_challenger_caption_validation_20170910/caption_validation_images_20170910' #验证集图片存放的位置
    else:
        out_im['filepath'] = 'ai_challenger_caption_train_20170902/caption_train_images_20170902' #训练集图片存放的位置
    out_s = []
    for a in annotsi:
        txt = []
        jimg = {}
        jimg['imgid'] = img['id']
        jimg['raw'] = a   # 原先的5个句子
        jimg['sentid'] = img['id']+'_'+str(sentid)
        for s in a['caption']:
            sent = list(jieba.cut(s))
            txt.append(sent)
            for word_uni in sent:
                raw_text_file.write(word_uni.encode("utf-8")+" ")
            raw_text_file.write("\n")

        jimg['tokens'] = txt #现在5个的句子的token组成的词拼接在一起
        jimg['sentids'] = []
        out_s.append(jimg)
    out_im['sentences'] = out_s
    out_im['split'] = split
    out.append(out_im)

    if i % 1000 == 0:
        print(str(i)+" / "+str(len(imgs)))

raw_text_file.close()
exit(0)
out_json['images']=out
out_json['dataset']='ai_challenger'
output_file = os.path.join('./json_preprocess_data', 'coco_ai_challenger_new_version.json')
json.dump(out_json, open(output_file, 'w'))


