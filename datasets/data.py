# -*- coding:utf-8 -*-
# from __future__ import print_function
import os
import sys
import time
import cv2
import random
import numpy as np
import json
import images as Images


class InData():
    def __init__(self, config, isTrain=True):
        self.config = config
        self.img_channels = 3
        self.path_index = 0

        dictfile = open(self.config['model']['dictionary']).readlines()
        self.classes = []
        for line in dictfile:
            self.classes.append(line.strip())
        self.classes_num = len(self.classes)
        print ("CLASSES Number:", self.classes_num)
        
        if isTrain:
            self.setname        = 'train'
            self.image_size     = self.config['train']['size']
            self.labelfile      = self.config['train']['labelfile']
            self.imgdir         = self.config['train']['imgdir']
            self.batch          = self.config['train']['batch']
            self.size           = self.config['train']['size']
            self.augment        = [['origin']]
            for i in self.config['train']['augment'].split(','):
                self.augment.append(i.strip().split('-'))
                assert self.augment[-1][0] in ["jitter", "intensity", "flip", "blur", "rotate", "noise"]
        else:
            self.setname        = 'test'
            self.image_size     = self.config['test']['size']
            self.labelfile      = self.config['test']['labelfile']
            self.imgdir         = self.config['test']['imgdir']
            self.batch          = self.config['test']['batch']
            self.size           = self.config['test']['size']
            self.augment        = []
        print ("Data Augment Method:", self.augment)


    def load_data(self):
        '''
        return [list]: datapath, labelID
        '''
        dataf = []
        labels = []
        data_list = open(self.labelfile).readlines()
        for ids, i in enumerate(data_list):
            filename, label_ = i.strip().split('\t')
            filedir = os.path.join(self.imgdir, filename)
            label_n = np.zeros(self.classes_num)
            label_n[self.classes.index(label_)] = 1.
            data_n = filedir
            labels.append(label_n)
            dataf.append(data_n)
            if ids % 10000 == 0:
                print (" "+str(ids+1)+" images had been loaded")

        labels = np.array(labels)
        dataf = np.array(dataf)
        print (' ',str(ids+1), ' image loaded success with total shape:', dataf.shape, labels.shape)
        return dataf, labels


    def prepare_data(self):
        print (" Loading data", self.setname, "......")
        self._datas, self._labels = self.load_data()
        print (" shape data | label:", np.shape(self._datas), np.shape(self._labels))
        self._list_len = np.shape(self._datas)[0]
        if self.setname == 'train':
            print (" Shuffling data ......")
            indices = np.random.permutation(len(self._datas))
            self._datas = self._datas[indices]
            self._labels = self._labels[indices]
        print (" Prepare Finished ......")


    def batch_read(self, augment_epoch=False):
        '''
        batch data generator, each epoch with different shuffled imagepath list
        : augment_epoch      shut off the augment method in some exact epoch
        '''
        batch_images = []
        
        ## prepare for each batch
        if self.path_index + self.batch > self._list_len:
            ## re-shuffle
            if self.setname == 'train':
                print (" Shuffling data ......")
                indices = np.random.permutation(len(self._datas))
                self._datas = self._datas[indices]
                self._labels = self._labels[indices]
            self.path_index = 0
        batch_paths = self._datas[self.path_index : self.path_index + self.batch]
        batch_labels = self._labels[self.path_index : self.path_index + self.batch]
        self.path_index += self.batch


        for i in batch_paths:
            _image = cv2.imread(i)

            if _image is None:
                print ("NoneTypeError: Image read in fail:", i)
            if len(_image.shape) != 3:
                print ("ShapeError: Image shape not equal to 3: ", i)
                continue
            if augment_epoch:
                method = np.random.choice(self.augment)
                if method[0] == 'jitter':
                    if len(method) > 1:
                        _image = cv2.resize(_image, (self.size, self.size))
                        image = Images.jitter(_image, ratio=float(method[1]))
                    else:
                        image = Images.jitter(_image)
                elif method[0] == 'blur':
                    if len(method) > 1:
                        image = Images.blur(_image, maxlevel=int(method[1]))
                    else:
                        image = Images.blur(_image)
                elif method[0] == 'rotate':
                    if len(method) > 1:
                        image = Images.rotate(_image, angle_bound=int(method[1]))
                    else:
                        image = Images.rotate(_image)
                elif method[0] == 'intensity':
                    image = Images.intensity(_image)
                elif method[0] == 'flip':
                    image = Images.flip(_image)
                elif method[0] == 'noise':
                    image = Images.noise(_image)
                else:
                    image = _image   ## origin
            else:
                image = _image
            try:
                image = cv2.resize(image, (self.size, self.size))
            except:
                print("AugmentError: augment method ERROR ", method, " ... Using origin format instead")
                image = cv2.resize(_image, (self.size, self.size))                
            batch_images.append(image)
        batch_images = np.array(batch_images)
        batch_images = batch_images.reshape([-1, self.img_channels, self.size, self.size])
        batch_images = batch_images.transpose([0, 2, 3, 1])
        return batch_images, batch_labels








