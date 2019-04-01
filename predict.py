from __future__ import print_function
import tensorflow as tf
# from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
import os, sys, cv2, json, time
from datasets.images import color_preprocessing
import argparse



class pred_classify():
    def __init__(self, config, args):
        self.config                            = config
        self.size                              = config['test']['size']
        self.channels                          = 3
        self.backbone                          = config['model']['backbones']

        ## gpu setting
        os.environ['CUDA_VISIBLE_DEVICES']     = '3'
        # if config['model']['gpus'] != ''
        #     self.gpu_set                       = config['model']['gpus'].split(',')
        #     self.gpu_num                       = len(self.gpu_set)
        #     if self.gpu_num > 1:
        #         gpu_id                         = self.gpu_set[0].strip()
        #     else:
        #         gpu_id                         = self.gpu_set
        #     os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        # else:
        #     self.gpu_num                       = 0


        ## define network 
        if self.backbone == 'SE_Inception_resnet_v2':
            from scripts.SE_Inception_resnet_v2 import SE_Inception_resnet_v2 as BackboneNet
        elif self.backbone == 'SE_Inception_v4':
            from scripts.SE_Inception_v4 import SE_Inception_v4 as BackboneNet
        elif self.backbone == 'SE_ResNeXt':
            from scripts.SE_ResNeXt import SE_ResNeXt as BackboneNet
        elif self.backbone == 'VGG16':
            from scripts.VGG16 import VGG16 as BackboneNet
        elif self.backbone == 'VGG19':
            from scripts.VGG19 import VGG19 as BackboneNet
        else:
            print("BackboneError: please use the defined backbone in ./scripts ......")


        ## training module
        # self.Net_graph = tf.Graph()
        # with self.Net_graph.as_default():
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as self.sess:
            with tf.device('/cpu:0'):
                with tf.variable_scope('cpu_vars'):
                    ## model ops input define
                    self.x = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channels])
                    self.training_flag = tf.placeholder(tf.bool)
                    self.model = BackboneNet(self.x, training=self.training_flag, config=self.config)

        ## checkpoint loader
        if args.model is None:
            self.checkpoint = tf.train.get_checkpoint_state(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model'))
            if self.checkpoint and tf.train.checkpoint_exists(self.checkpoint.model_checkpoint_path):
                self.checkpoint = self.checkpoint.model_checkpoint_path
        else:
            self.checkpoint = args.model
 
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())
        self.saver.restore(self.sess, self.checkpoint)


    def Evaluate(self, img):
        img = cv2.resize(img, (self.size, self.size))
        img = img.reshape([-1, self.channels, self.size, self.size])
        img = img.transpose([0, 2, 3, 1])
        img = color_preprocessing(img)

        test_feed = {
            self.x: img,
            self.training_flag:False,
        }

        self.logit = self.model.model
        self.conf = self.sess.run([self.logit], feed_dict=test_feed)
        self.ans = np.argmax(self.conf)
        
        return self.ans



if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='Experiment Platform for Image Recognition')
    argparser.add_argument('-c', '--config',      help='path to configuration file')
    argparser.add_argument('-i', '--imgdir',      help='imageset txtfile with classes or image directory')
    argparser.add_argument('-o', '--outpath',     help='outfile path')
    argparser.add_argument('-m', '--model',       default=None,  help='model path')
    args = argparser.parse_args()

    with open(args.config) as config_buffer:
        config = json.loads(config_buffer.read())

    classes = []
    clsdict = {}
    for i in open(config['model']['dictionary']).readlines():
        cate = i.strip()
        if cate not in classes:
            classes.append(cate)
            clsdict[cate] = [0,0]     ## [ clsdict[cate][0], clsdict[cate][1] ] = [ #result, #pred ]
    clsdict['other'] = [0,0]

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    class_num = len(classes)
    recognize = pred_classify(config, args)

    if args.imgdir.find('txt') == -1:
        imglist = os.listdir(args.imgdir)
        for i in imglist:
            imgpath = os.path.join(args.imgdir, i)
            img = cv2.imread(imgpath)
            if len(img.shape) != 3:
                print("Error in reading image: ", imgpath)
            pred = recognize.Evaluate(img)
            label = os.path.basename(imgpath).split('_')[-1][:-4]

            if label not in classes:
                clsdict['other'][1] += 1
                print("LabelError: an `outbound` label appear ......", i, label)
            else:
                clsdict[label][0] += 1
                if classes[pred] == label:
                    clsdict[classes[pred]][1]  += 1 
                print(imgpath, "Label: ", label, " Prediction: ", classes[pred])

    else:
        imgfile = open(args.imgdir).readlines()
        for line in imgfile:
            imgpath, label = line.strip().split('\t')
            # img = cv2.imread(imgpath)
            img = cv2.imread('/data/'+imgpath)
            if len(img.shape) != 3:
                print("Error in reading image: ", imgpath)
            pred = recognize.Evaluate(img)        

            if label not in classes:
                clsdict['other'][0] += 1
                print("LabelError: an `outbound` label appear ......", i, label)
            else:
                clsdict[label][0] += 1
                if classes[pred] == label:
                    clsdict[classes[pred]][1] += 1
                print(imgpath, "Label: ", label, " Prediction: ", classes[pred])


    outfile = open(args.outpath, 'w')
    label_sum = 0
    pred_sum = 0
    for cate in sorted(clsdict.keys()):
        label_sum += clsdict[cate][0]
        pred_sum  += clsdict[cate][1]
        if clsdict[cate][0] == 0:
            outfile.write('{}\tlabel_count: {}\tpred_count: {}\t class_acc: {}\n'.format( cate, clsdict[cate][0], clsdict[cate][1], '0' ))
        else:
            outfile.write('{}\tlabel_count: {}\tpred_count: {}\t class_acc: {}\n'.format( cate, clsdict[cate][0], clsdict[cate][1], float(clsdict[cate][1])/clsdict[cate][0] ))
    if label_sum != 0:
        outfile.write('{}\tlabel_count: {}\tpred_count: {}\t class_acc: {}\n'.format('Total', label_sum, pred_sum, float(pred_sum)/label_sum ))
    else:
        outfile.write('{}\tlabel_count: {}\tpred_count: {}\t class_acc: {}\n'.format('Total', label_sum, pred_sum, '0' ))





