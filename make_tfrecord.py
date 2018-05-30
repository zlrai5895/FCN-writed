#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:29:51 2018

@author: zhanglei
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from scipy import misc
import time



def data_read(flag,temp_dir):
    img_dir='/home/zhanglei/tensorflow_practice/writed/ADEChallengeData2016/'+flag+'/'+temp_dir+'/'
    img_names=os.listdir(img_dir)
    img_path=sorted([img_dir + x for x in img_names])
    
    return img_path



train_img_path=data_read('images','training')
train_gt_path=data_read('annotations','training')

val_img_path=data_read('images','validation')
val_gt_path=data_read('annotations','validation')





def create_record(files_name,labels_name,name):
    writer=tf.python_io.TFRecordWriter(name+'.tfrecord')
    for i in range(len(files_name)):
        img=misc.imread(files_name[i])
        img=misc.imresize(img, (224, 224))
        img=img.astype(np.uint8)
        img_raw=img.tobytes()                   #将图片转化为原生bytes
        
        gt=misc.imread(labels_name[i])
        gt=misc.imresize(gt, (224, 224))
        gt=gt.astype(np.uint8)
        gt_raw=gt.tobytes()                   #将图片转化为原生bytes
        
        example=tf.train.Example(features=tf.train.Features(feature={
                'imgs': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[gt_raw]))}))#example:img+label
        serialized=example.SerializeToString()   #序列化
        writer.write(serialized)    ##写入文件
        
        if i %200==0:
            print(i,len(files_name),img.shape,gt.shape)
        
    writer.close()
    
    
def read_record(filename,h,w):
    filename_quene=tf.train.string_input_producer([filename],shuffle=False)
   
    train_reader=tf.TFRecordReader()
    _,serialized_example=train_reader.read(filename_quene)
    
    
    features=tf.parse_single_example(serialized_example,features={
            'imgs': tf.FixedLenFeature([],tf.string),
            'label': tf.FixedLenFeature([],tf.string) })
    
    img=tf.decode_raw(features['imgs'],tf.uint8)
    img=tf.reshape(img,[h,w,3])
    
    label=tf.decode_raw(features['label'],tf.uint8)
    label=tf.reshape(label,[h,w])
    
    return img,label



    
#create_record(train_img_path,train_gt_path,'ade_train')
train_img,train_label=read_record('train.tfrecord',224,224)
    

#create_record(val_img_path,val_gt_path,'val')
#val_img,val_label=read_record('val.tfrecord',224,224)


train_img_batch,train_label_batch=tf.train.batch([train_img,train_label],batch_size=10,capacity=200,num_threads=6)

init=tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    coord=tf.train.Coordinator()   #创建一个协调器，管理线程
    threads=tf.train.start_queue_runners(sess=sess,coord=coord) #启动QueueRunner, 此时文件名队列已经进队
    for i in range(30000):
        imgs,labels=sess.run([train_img_batch,train_label_batch])
        print(imgs)
        print(labels)
        print('----------------------------',i,'--------------------------')
    coord.request_stop()
    coord.join(threads)





