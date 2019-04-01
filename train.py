from __future__ import print_function
import json, os, sys, cv2, time
import numpy as np
import tensorflow as tf
import argparse
from datasets.data import InData
from datasets.images import color_preprocessing
from utils.ckpt import partial_restore_checkpoint, average_gradients
from tensorflow.python import debug as tf_debug



argparser = argparse.ArgumentParser(description='Experiment Platform for Image Recognition')
argparser.add_argument('-c', '--config', help='path to configuration file')  
args = argparser.parse_args()


## config arguments load in 
with open(args.config) as config_buffer:
    config = json.loads(config_buffer.read())


project_name                           = config['train']['name']
backbone                               = config['model']['backbones']
init_learning_rate                     = config['model']['learning_rate']
lrs                                    = config['model']['lr_schedule']
total_epochs                           = config['model']['max_epoch']
batch_size                             = config['train']['batch']
image_size                             = config['train']['size']
test_batch                             = config['test']['batch']
augment_warmup                         = config['train']['augment_warmup']
train_iteration                        = len(open(config['train']['labelfile']).readlines()) //  batch_size
test_iteration                         = len(open(config['test']['labelfile']).readlines()) //  test_batch
weight_decay                           = config['model']['weight_decay']
momentum                               = config['model']['momentum']
if config['model']['gpus'] != '':
    os.environ["CUDA_VISIBLE_DEVICES"] = config['model']['gpus']
    gpu_set                            = config['model']['gpus'].split(',')
    gpu_num                            = len(gpu_set)
else:
    gpu_num                            = 0

if backbone == 'SE_Inception_resnet_v2':
    from scripts.SE_Inception_resnet_v2 import SE_Inception_resnet_v2 as BackboneNet
elif backbone == 'SE_Inception_v4':
    from scripts.SE_Inception_v4 import SE_Inception_v4 as BackboneNet
elif backbone == 'SE_ResNeXt':
    from scripts.SE_ResNeXt import SE_ResNeXt as BackboneNet
elif backbone == 'VGG16':
    from scripts.VGG16 import VGG16 as BackboneNet
elif backbone == 'VGG19':
    from scripts.VGG19 import VGG19 as BackboneNet
else:
    print("BackboneError: please use the defined backbone in ./scripts ......")

## logfile
localt = time.localtime()
filename = 'log-{:0>4}{:0>2}{:0>2}-{:0>2}{:0>2}{:0>2}.txt'.format(localt[0], localt[1], localt[2], localt[3], localt[4], localt[5])
logsfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs', filename)


## trainpaths, testpaths prepared
trainset = InData(config=config, isTrain=True)
testset = InData(config=config, isTrain=False)
trainset.prepare_data()
testset.prepare_data()


## training module
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    with tf.device('/cpu:0'):

        ## initialize CPU variables
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)   ## SGD-Momentum
        reuse = False
        _hub_model = []

        if gpu_num == 0:
            ## allocate model-variables in CPUs
            print("Using CPU only")
            with tf.variable_scope('cpu_vars'):
                ## model ops input define
                x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3])
                labels = tf.placeholder(tf.float32, shape=[None, trainset.classes_num])
                training_flag = tf.placeholder(tf.bool)

                ## model & loss 
                logits = BackboneNet(x, training=training_flag, config=config, reuse=reuse).model

                ## cost with L2 norm
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
                l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])                 ## L2-norm
                grad = optimizer.compute_gradients(cost + l2_loss * weight_decay)
                _hub_model.append([x, labels, learning_rate, training_flag, logits, labels, cost, grad])
        else:
            ## allocate model-variables in GPUs
            for gpu_id in gpu_set:
                gpu_id = int(gpu_id.strip())
                with tf.device('/gpu:%d' % gpu_id):
                    print("Using Multiple GPUs: tower_%d" % gpu_id)
                    with tf.name_scope('tower_%d' % gpu_id):
                        with tf.variable_scope('cpu_vars'):
                            ## model ops input define
                            x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3])
                            labels = tf.placeholder(tf.float32, shape=[None, trainset.classes_num])
                            training_flag = tf.placeholder(tf.bool)

                            ## model & loss 
                            logits = BackboneNet(x, training=training_flag, config=config, reuse=reuse).model
                            reuse = True

                            ## cost with L2 norm
                            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
                            l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])                 ## L2-norm
                            grad = optimizer.compute_gradients(cost + l2_loss * weight_decay)
                            _hub_model.append([x, labels, learning_rate, training_flag, logits, labels, cost, grad])

        ## Summarize GPUs result
        tower_x, tower_labels, tower_lrs, tower_tfs, tower_logits, tower_labels, tower_costs, tower_grads = zip(*_hub_model)

        total_cost = tf.add_n(tower_costs)
        apply_gradients_op = optimizer.apply_gradients(average_gradients(tower_grads), global_step=global_step)
        total_logits = tf.concat(tower_logits, 0)
        total_labels = tf.concat(tower_labels, 0)

        correct_prediction = tf.equal(tf.argmax(total_logits, 1), tf.argmax(total_labels, 1))
        correct_count = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        ## pretrained model
        _epoch = 0
        if config['train']['pretrained'] != '':
            saver, inner_step = partial_restore_checkpoint(config['train']['pretrained'], sess, global_step)
            _epoch = int(os.path.basename(config['train']['pretrained']).split('.')[0].split('_')[-1].split('-')[-1])
        else:
            ckpt = tf.train.get_checkpoint_state(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model'))
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver, inner_step = partial_restore_checkpoint(ckpt.model_checkpoint_path, sess, global_step)
                _epoch = int(os.path.basename(ckpt.model_checkpoint_path).split('.')[0].split('_')[-1].split('-')[-1])
            else:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
                inner_step = sess.run([global_step])

        ## tensorboard 
        train_writer = tf.summary.FileWriter('logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('logs/test')

        ## lr_schedule
        epoch_learning_rate = init_learning_rate
        lr_schedule = [[], []]
        if lrs == '':
            for i in range(30, total_epochs, 30):
                lr_schedule[0].append(i)
                lr_schedule[1].append(0.1)
        else:
            try:
                _lrs = lrs.split(',')
                for i in range(0, len(_lrs), 2):
                    lr_schedule[0].append(int(_lrs[i].strip()))
                    lr_schedule[1].append(float(_lrs[i+1].strip()))
                if epoch in lr_schedule[0]:
                    epoch_learning_rate = epoch_learning_rate * lr_schedule[1][lr_schedule.index(epoch)]
                else:
                    epoch_learning_rate = epoch_learning_rate
            except:
                print("ListError: lr_schedule load in fail, please check the config. No LR decay settings right now ......")
        print("Learning Rate Schedule:\tInitial-LR:", init_learning_rate, "\tSchedule [epoch,ratio]:", lr_schedule)

	## tensorflow debugger
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        ## start training
        for epoch in range(_epoch+1, total_epochs+1):
            print('Start Epoch ', epoch, "......")

            ##  init learning rate by lr_schedule and #epoch
            if epoch in lr_schedule[0]:
                post_lr = epoch_learning_rate
                epoch_learning_rate = epoch_learning_rate * lr_schedule[1][lr_schedule[0].index(epoch)]
                print(" >>>>> Epoch ", epoch, "\tLearning Rate:", post_lr, "==>>", epoch_learning_rate)


            ## start training
            train_cc = 0.0
            train_acc = 0.0
            train_loss = 0.0
            start_time = 0.0
            
            if gpu_num == 0:
                iter_step = 1
            else:
                iter_step = gpu_num
            
            for outer_step in range(1, train_iteration + 1, iter_step):
                if (outer_step-1) % (3*iter_step) == 0:
                    end_time = time.time()
                    print('...... Step ', outer_step-1, '\tAvg_Loss: ', round(train_loss/(outer_step*batch_size), 8), '\tSpeed: ', round((3*iter_step * batch_size) / (end_time-start_time), 2), 'sample / sec' )
                    start_time = time.time()

                ## feed each graph
                train_feed_dict = {}
                for _model_ops in _hub_model:
                    _x, _labels, _learning_rate, _training_flag, _, _, _, _ = _model_ops    
                    if epoch > augment_warmup:
                        batch_x, batch_y = trainset.batch_read(augment_epoch=True)
                    else:
                        batch_x, batch_y = trainset.batch_read(augment_epoch=False)
                    batch_x = color_preprocessing(batch_x)

                    train_feed_dict[_x]             = batch_x
                    train_feed_dict[_labels]        = batch_y
                    train_feed_dict[_learning_rate] = epoch_learning_rate
                    train_feed_dict[_training_flag] = True

                _, batch_loss = sess.run([apply_gradients_op, total_cost], feed_dict=train_feed_dict)
                batch_correct_num = correct_count.eval(feed_dict=train_feed_dict)
                # _, batch_loss, batch_correct_num = sess.run([apply_gradients_op, total_cost, correct_count], feed_dict=train_feed_dict)
                # print('train_batch_loss', batch_loss)
                train_loss += batch_loss 
                train_acc += batch_correct_num

            train_loss /= train_iteration                          # average loss
            train_acc /= (train_iteration*batch_size)              # average accuracy

            ## train ops
            train_summary = tf.Summary(value=[tf.Summary.Value(tag='Loss', simple_value=train_loss),
                                              tf.Summary.Value(tag='Accuracy', simple_value=train_acc),
                                              tf.Summary.Value(tag='learning_rate', simple_value=epoch_learning_rate)])
            train_writer.add_summary(summary=train_summary, global_step=epoch)


            ## eval ops
            print('Evaluate ......')
            test_loss = 0.0
            test_acc = 0.0
            test_feed_dict = {}
            for _ in range(0, test_iteration, iter_step):
                for _model_ops in _hub_model:
                    _x, _labels, _learning_rate, _training_flag, _, _, _, _ = _model_ops    
                    batch_x, batch_y = testset.batch_read(augment_epoch=False)
                    batch_x = color_preprocessing(batch_x)

                    test_feed_dict[_x]             = batch_x
                    test_feed_dict[_labels]        = batch_y
                    test_feed_dict[_learning_rate] = epoch_learning_rate
                    test_feed_dict[_training_flag] = False

                test_batch_loss, test_batch_correct_num = sess.run([total_cost, correct_count], feed_dict=test_feed_dict)
                test_loss += test_batch_loss
                test_acc += test_batch_correct_num
                # print('test_batch_loss', test_batch_loss)
            test_loss /= test_iteration
            test_acc /= (test_iteration*test_batch)

            test_summary = tf.Summary(value=[tf.Summary.Value(tag='Loss', simple_value=test_loss),
                                              tf.Summary.Value(tag='Accuracy', simple_value=test_acc)])
            test_writer.add_summary(summary=test_summary, global_step=epoch)


            line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f \n" % (
                epoch, total_epochs, train_loss, train_acc, test_loss, test_acc)
            print(line)

            with open(logsfile, 'a') as f:
                f.write(line)

            if project_name == '':
                saver.save(sess=sess, save_path=os.path.join(os.path.dirname(os.path.realpath(__file__)),'model', str(backbone)+'_Epoch-'+str(epoch)+'.ckpt'))
            else:
                saver.save(sess=sess, save_path=os.path.join(os.path.dirname(os.path.realpath(__file__)),'model', str(project_name)+'_Epoch-'+str(epoch)+'.ckpt'))







