#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 10:49:34 2018

@author: zhanglei
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:29:51 2018

@author: zhanglei
"""
import numpy as np
import tensorflow as tf
import scipy.io as scio
from scipy import misc
import sys
import logging
import datetime


FLAGS=tf.flags.FLAGS
tf.flags.DEFINE_integer('batchsize','10','trainning batchsize')#flag
tf.flags.DEFINE_float('learning_rate','1e-4','learning_rate')#flag
tf.flags.DEFINE_bool('reuse', "False", "reuse the pretrained model")
tf.flags.DEFINE_bool('train', "True", "train or test")
tf.flags.DEFINE_string('checkpoint', "checkpoint", "dir to save model")
tf.flags.DEFINE_string('log', "log", "dir to summary")


IMAGE_SIZE=224
NUM_OF_CLASSESS = 151
NUM_EPOCHES=100001


def initLogging(logFilename='record.log'):
  """Init for logging
  """
  logging.basicConfig(
                    level= logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)
  
  
initLogging()


    
    
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



train_img,train_label=read_record('train.tfrecord',224,224)   
train_img_batch,train_label_batch=tf.train.batch([train_img,train_label],batch_size=10,capacity=200,num_threads=6)
train_label_batch=tf.expand_dims(train_label_batch,-1)#expand dim
    
val_img,val_label=read_record('val.tfrecord',224,224)   
val_img_batch,val_label_batch=tf.train.batch([val_img,val_label],batch_size=10,capacity=200,num_threads=6)
val_label_batch=tf.expand_dims(val_label_batch,-1)#expand dim



##########################pretrained vgg19#####################################   
pre_train_model_data=scio.loadmat('imagenet-vgg-verydeep-19.mat')
weights=weights = np.squeeze(pre_train_model_data['layers'])  #squeeze


layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )
weight_list=[]
bias_list=[]
for i, name in enumerate(layers):
    if name[:4]=='conv':
        kernels, bias = weights[i][0][0][0][0]
        weight_list.append(np.transpose(kernels,axes=(1, 0, 2, 3)))
        bias_list.append(bias.reshape(-1))


def conv_bias_relu(input_tensor,scope_name,ind):
    with tf.variable_scope(scope_name):
        init=tf.constant_initializer(value=weight_list[ind],dtype=tf.float32)
        kernel=tf.get_variable('kernel',shape=weight_list[ind].shape,dtype=tf.float32,initializer=init)
        conv=tf.nn.conv2d(input_tensor,kernel,[1,1,1,1],padding='SAME',name='conv')
        bias=tf.add(conv,bias_list[ind],name='bias')
        relu=tf.nn.relu(bias,name='relu')
    return relu

def average_pool(input_tensor,scope_name):
    with tf.variable_scope(scope_name):
        pool=tf.nn.avg_pool(input_tensor,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME',name='pool')
    return pool

def conv_bias_relu_init(input_tensor,scope_name,shape):
    with tf.variable_scope(scope_name):
        kernel=tf.Variable(tf.truncated_normal(shape=shape,stddev=0.02))
        conv=tf.nn.conv2d(input_tensor,kernel,[1,1,1,1],padding='SAME',name='conv')
        bias=tf.add(conv,tf.constant(0,dtype=tf.float32,shape=[shape[-1]]),name='bias')
        relu=tf.nn.relu(bias,name='relu')
    return relu

def deconv(input_tensor,scope_name,shape,out_shape,strider=2):
    with tf.variable_scope(scope_name) :#tf.variable_scope(scope_name)
        kernel=tf.Variable(tf.truncated_normal(shape=shape,stddev=0.02))
        deconv=tf.nn.conv2d_transpose(input_tensor,kernel,out_shape,[1,strider,strider,1],padding='SAME')
        bias=tf.add(deconv,tf.constant(0,dtype=tf.float32,shape=[deconv.shape[3].value]),name='bias')    
    return bias    #tf.nn.conv2d_transpose




def summary_activition(var):
    tf.summary.histogram(var.op.name + "/activation", var)
    tf.summary.scalar(var.op.name+'/sparsity', tf.nn.zero_fraction(var))#返回0在value中的小数比例


#########################net struct###########################
def fcn_net(img,gt,keep_prob):
    relu1_1=conv_bias_relu(img,'conv1_1',0)
    summary_activition(relu1_1)
    relu1_2=conv_bias_relu(relu1_1,'conv1_2',1) 
    summary_activition(relu1_2)
    pool1=average_pool(relu1_2,'pool1')
    
    relu2_1=conv_bias_relu(pool1,'conv2_1',2)
    relu2_2=conv_bias_relu(relu2_1,'conv2_2',3)
    pool2=average_pool(relu2_2,'pool2')
    
    relu3_1=conv_bias_relu(pool2,'conv3_1',4)
    summary_activition(relu3_1)
    relu3_2=conv_bias_relu(relu3_1,'conv3_2',5) 
    summary_activition(relu3_2)
    
    
    relu3_3=conv_bias_relu(relu3_2,'conv3_3',6)
    summary_activition(relu3_3)
    relu3_4=conv_bias_relu(relu3_3,'conv3_4',7) 
    summary_activition(relu3_4)
    pool3=average_pool(relu3_4,'pool3')
    
    relu4_1=conv_bias_relu(pool3,'conv4_1',8)
    summary_activition(relu4_1)
    relu4_2=conv_bias_relu(relu4_1,'conv4_2',9) 
    summary_activition(relu4_2)
    
    relu4_3=conv_bias_relu(relu4_2,'conv4_3',10)
    summary_activition(relu4_3)
    relu4_4=conv_bias_relu(relu4_3,'conv4_4',11) 
    summary_activition(relu4_4)
    pool4=average_pool(relu4_4,'pool4')
    
    
    relu5_1=conv_bias_relu(pool4,'conv5_1',12)
    summary_activition(relu5_1)
    relu5_2=conv_bias_relu(relu5_1,'conv5_2',13) 
    summary_activition(relu5_2)
    
    relu5_3=conv_bias_relu(relu5_2,'conv5_3',14)
    summary_activition(relu5_3)
    relu5_4=conv_bias_relu(relu5_3,'conv5_4',15) 
    summary_activition(relu5_4)
    pool5=average_pool(relu5_4,'pool5')
    
    relu6=conv_bias_relu_init(pool5,'conv6',[7, 7, 512, 4096])
    summary_activition(relu6)
    relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
    
    relu7=conv_bias_relu_init(relu_dropout6,'conv7',[1, 1, 4096, 4096])
    summary_activition(relu7)
    relu_dropout7 = tf.nn.dropout(relu6, keep_prob=keep_prob)
    
    
    #kernel=tf.Variable(tf.truncated_normal(shape=[1, 1, 4096, NUM_OF_CLASSESS],stddev=0.02))
    #conv=tf.nn.conv2d(relu_dropout7,kernel,[1,1,1,1],padding='SAME',name='conv8')
    #bias8=tf.add(conv,tf.constant(0,dtype=tf.float32,shape=[NUM_OF_CLASSESS]),name='bias')
    
    
    
    deconv1=deconv(relu_dropout7,'deconv1',[4,4,pool4.shape[3].value,4096],tf.shape(pool4))
    fuse_1 = tf.add(deconv1, pool4, name="fuse_1")
    
    
    deconv2=deconv(fuse_1,'deconv2',[4,4,pool3.shape[3].value,pool4.shape[3].value],tf.shape(pool3))###.value
    fuse_2 = tf.add(deconv2, pool3, name="fuse_2")
    
    shape = tf.shape(img)
    deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
    deconv3=deconv(fuse_2,'decon3',[16,16,NUM_OF_CLASSESS,pool3.shape[3].value],deconv_shape3,strider=8)
    
    pre=tf.argmax(deconv3,axis=-1)
    pre=tf.expand_dims(pre,-1)#expand dim
    
    
    #############visualize##############
    tf.summary.image("input_image", img, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(gt, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pre, tf.uint8), max_outputs=2)

    return pre,deconv3

def pre_process(imgs):
    processed_img_batch=imgs-np.mean(imgs, axis = 0)
    processed_img_batch/= np.std(processed_img_batch, axis = 0)
    
    return processed_img_batch

def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = np.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()




def train():

    img=tf.placeholder(dtype=tf.float32,shape=[None,IMAGE_SIZE,IMAGE_SIZE,3])
    gt=tf.placeholder(dtype=tf.int32,shape=[None,IMAGE_SIZE,IMAGE_SIZE,1])
    keep_prob=tf.placeholder(dtype=tf.float32)
    pre,deconv3=fcn_net(img,gt,keep_prob)
    ###############loss calculate####################
    loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=deconv3,labels=tf.squeeze(gt,squeeze_dims=3)))
    tf.summary.scalar('loss',loss)
    ##############optimizer##########################
    op=tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)  #FLAGS.learning_rate
    merged =tf.summary.merge_all()
    
    
    ##############initialize####################
    init=tf.global_variables_initializer()
    
    
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(FLAGS.log + '/train', sess.graph)
        val_writer = tf.summary.FileWriter(FLAGS.log + '/val')
        saver=tf.train.Saver(max_to_keep=5)##
        if FLAGS.reuse==False:
            sess.run(init)    #init 
        else:
#            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint)   #pre_trained
#            saver.restore(sess,ckpt.model_checkpoint_path)  #restore
             saver.restore(sess, 'checkpoint/model.ckpt-1220')
        coord=tf.train.Coordinator()   #创建一个协调器，管理线程
        threads=tf.train.start_queue_runners(sess=sess,coord=coord) #启动QueueRunner, 此时文件名队列已经进队
        try:
            for ep in range(1,NUM_EPOCHES):
                temp_train_img,temp_train_label=sess.run([train_img_batch,train_label_batch])
                processed_img_batch=pre_process(temp_train_img)
                _,summary,predicts_value,train_loss=sess.run([op,merged,pre,loss],feed_dict={img:processed_img_batch,gt:temp_train_label,keep_prob:0.85})
                train_precision = np.mean(predicts_value==temp_train_label)
                train_writer.add_summary(summary,ep)
    #            print('epoch '+str(ep)+':  ',train_loss) 
                #print('----------------------------',i,'--------------------------')
                count = ep % 500
                mes=('>>Step: %d loss = %.4f acc = %.3f'% (ep,train_loss, train_precision))
                view_bar(mes, count, 500)
                if ep%500==0 and ep!=0:
                    temp_val_img,temp_val_label=sess.run([val_img_batch,val_label_batch])
                    processed_val_img_batch=pre_process(temp_val_img)
                    _,summary,val_predicts,val_loss=sess.run([op,merged,pre,loss],feed_dict={img:processed_val_img_batch,gt:temp_val_label,keep_prob:0.85})
                    val_precision = np.mean(val_predicts==temp_val_label)
                    val_writer.add_summary(summary,ep)
                    print('loss on val:',val_loss)
                    print('accuracy on val ',val_precision)
                    logging.info('>>%s Saving in %s' % (datetime.datetime.now(), FLAGS.checkpoint))
                    saver.save(sess,FLAGS.checkpoint+'/model.ckpt',ep)
                else:
                    if ep%100==99:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        summary, _ = sess.run([merged, op],feed_dict={img:processed_img_batch,gt:sess.run(train_label_batch),keep_prob:0.85},options=run_options,run_metadata=run_metadata)
                        train_writer.add_run_metadata(run_metadata, 'step%03d' % ep)
            train_writer.close()
            val_writer.close()
            coord.request_stop()
            coord.join(threads)
        except KeyboardInterrupt :
                print('Being interrupted')
                logging.info('>>%s Saving in %s' % (datetime.datetime.now(), FLAGS.checkpoint))
                saver.save(sess,FLAGS.checkpoint+'/model.ckpt',ep)
        finally:
                train_writer.close()
                val_writer.close()
                coord.request_stop()
                coord.join(threads)

def save_result(pre_batch,gt_batch,ep):
    img_shape=gt_batch.shape
    for i in range(img_shape[0]):
        temp_pre=pre_batch[i].reshape([224,224])
        misc.imsave('test_result/pre/pre'+str(ep*FLAGS.batchsize+i)+'.png',temp_pre)
        temp_gt=gt_batch[i].reshape([224,224])
        misc.imsave('test_result/gt/gt'+str(ep*FLAGS.batchsize+i)+'.png',temp_gt)


def test():
    img=tf.placeholder(dtype=tf.float32,shape=[None,IMAGE_SIZE,IMAGE_SIZE,3])
    gt=tf.placeholder(dtype=tf.int32,shape=[None,IMAGE_SIZE,IMAGE_SIZE,1])
    keep_prob=tf.placeholder(dtype=tf.float32)
    pre,deconv3=fcn_net(img,gt,keep_prob)
    ###############loss calculate####################
    loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=deconv3,labels=tf.squeeze(gt,squeeze_dims=3)))
    ##############optimizer##########################
    op=tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)  #FLAGS.learning_rate
    
    
    ##############initialize####################
    #init=tf.global_variables_initializer()
    with tf.Session() as sess:
        saver=tf.train.Saver()#
        #ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint)   #pre_trained
        #saver.restore(sess,ckpt.model_checkpoint_path)  #restore
        saver.restore(sess, 'checkpoint/model.ckpt-1220')
        
        coord=tf.train.Coordinator()   #创建一个协调器，管理线程
        threads=tf.train.start_queue_runners(sess=sess,coord=coord) #启动QueueRunner, 此时文件名队列已经进队
        test_precison=0
        for ep in range(NUM_EPOCHES):
            temp_test_img,temp_test_label=sess.run([val_img_batch,val_label_batch])
            processed_test_img_batch=pre_process(temp_test_img)
            _,predict_label=sess.run([op,pre],feed_dict={img:processed_test_img_batch,gt:temp_test_label,keep_prob:1.0})
            save_result(predict_label,temp_test_label,ep)
            temp_test_precision = np.mean(predict_label==temp_test_label)
            test_precison=(test_precison*ep+temp_test_precision)/(ep+1)
            print('------------',ep*FLAGS.batchsize,'-------------')
            print('current precision is : ',test_precison)
        coord.request_stop()
        coord.join(threads)





def main(argv=None):
    if FLAGS.train==True:
        train()
    else:
        test()

if __name__=='__main__':
    tf.app.run()