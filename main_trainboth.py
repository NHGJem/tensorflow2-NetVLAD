import numpy as np
import pandas as pd 
import os
from time import time 
import datetime

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Lambda
from tensorflow.math import l2_normalize

#import data_pipeline as pipe
import data_pipeline_hard_neg as pipe
import embedding_generator as emb
import netvlad_layer as netvlad

# =============================================================================
#
# Run this script to perform training on either Paris or Oxford dataset.
# Set up configs below before starting
# 
# This script will output logs for Tensorboard, weights, and embeddings.
# Weight checkpoints are generated on the first and last epochs, and also every 20 epochs after the first. Embeddings are overwritten.
#
# =============================================================================

#------------------------------------------------------------------------------
##################### Important configurations #####################

WHICH_DATASET = 'both' # only 'both' is a valid setting

# Path to images
OXFORD_PATH = os.path.join('..','data','oxbuild_images_zipped','oxbuild_images')
PARIS_PATH = os.path.join('..','data','paris_zipped','paris')

# Path to groundtruth txt files
OXFORD_PATH_TXT = os.path.join(OXFORD_PATH, 'gt_files_170407')
PARIS_PATH_TXT = os.path.join(PARIS_PATH, 'paris_120310')

# Path to save test embeddings
PARIS_EMBEDS = os.path.join('..','outputs','paris_embed')
OXFORD_EMBEDS = os.path.join('..','outputs','oxford_embed')

# Path to load VGG16 weights trained on ImageNet
#BASE_WEIGHT_PATH = os.path.join('..','data','weights','vgg16_weights_notop.h5')
# Which base to use: VGG16, MobileNetV2, or ResNet50V2 (case-sensitive)
BASE_MODEL = 'VGG16'
k_value = 64                    # VGG's settings are 64 and 4096. To keep the number of parameters relative,
embed_dimension = 4096          # divide one of these by a factor of 2 for MobileNet, and a factor of 4 (or both at factor of 2) for ResNet50.

# Use weights from previously trained NetVLAD models (e.g. continuing training)
USE_TRAINED_WEIGHTS = False
TRAIN_CONV5 = False              # Train last conv layer of VGG16 too, or train only NetVLAD layers. Leave as False if not using VGG16!
TRAINED_WEIGHTS_PATH = '...'      # If USE_TRAINED_WEIGHTS = True, put path to weights here   

# Save output logs (for Tensorboard) and weights
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOGPATH = os.path.join('..','outputs','logs',WHICH_DATASET,current_time)
OUTPUT_WEIGHT_PATH = os.path.join('..','outputs','weights',WHICH_DATASET)

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
####### Less important configs, can be left default or finetuned #######

TRIPLET_SIZE = 16     # Number of triplets in one minibatch
BATCH_SIZE = 20       # Number of minibatches in one epoch
VAL_BATCH_SIZE = 32   # Number of images to generate embeddings of at one time

IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

LEARNING_RATE = 0.001
VARIABLE_LR = True    # If true, will use Cosine LR Decay with Repeats
LOSS_MARGIN = 0.1     # Margin for triplet loss

# Epoch settings
STARTING_EPOCH = 0    # set to 0 if it's a fresh training, set to the last epoch number of previous training if continued
EPOCHS = 200          # training will proceed until epoch no. STARTING_EPOCH+EPOCHS-1

paris_labels = ['defense','eiffel','invalides','louvre','moulinrouge','museedorsay','notredame','pantheon','pompidou','sacrecoeur','triomphe']
oxford_labels = ['all_souls','ashmolean','balliol','bodleian','christ_church','cornmarket','hertford','keble','magdalen','pitt_rivers','radcliffe_camera']

# Dataset normalization
paris_mean = tf.convert_to_tensor([0.44528243,0.4364394,0.41679484])
paris_std = tf.convert_to_tensor([0.13248205,0.14063343,0.16034052])

oxford_mean = tf.convert_to_tensor([0.43960404, 0.42639694, 0.38630432])
oxford_std = tf.convert_to_tensor([0.12214108, 0.12625049, 0.13406576])

tuple_of_types = (tuple([tf.string]*3), tf.bool, tf.float32, tf.float32, tf.int32)

#------------------------------------------------------------------------------

# ============================================================================= 

### Ignore warnings -----------------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
import logging
tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
### ---------------------------------------------------------------------------

### Write Tensorboard summaries -----------------------------------------------
train_summary_writer = tf.summary.create_file_writer(os.path.join(LOGPATH,"train"))
validation_summary_writer = tf.summary.create_file_writer(os.path.join(LOGPATH,"validation"))
test_summary_writer = tf.summary.create_file_writer(os.path.join(LOGPATH,"test"))
### ---------------------------------------------------------------------------

### Set up dataset depending on Paris or Oxford -------------------------------
if WHICH_DATASET == 'both':
    paris_query_list = pipe.generate_query_list(PARIS_PATH_TXT)
    paris_training_x = tf.data.Dataset.from_generator(pipe.dataset_maker, tuple_of_types, args = (PARIS_PATH_TXT,paris_query_list,True, paris_mean, paris_std, IMG_SIZE))
    paris_validation_x = tf.data.Dataset.from_generator(pipe.dataset_maker, tuple_of_types, args = (PARIS_PATH_TXT,paris_query_list,False, paris_mean, paris_std, IMG_SIZE))
    oxford_query_list = pipe.generate_query_list(OXFORD_PATH_TXT)
    oxford_training_x = tf.data.Dataset.from_generator(pipe.dataset_maker, tuple_of_types, args = (OXFORD_PATH_TXT,oxford_query_list,True, oxford_mean, oxford_std, IMG_SIZE))
    oxford_validation_x = tf.data.Dataset.from_generator(pipe.dataset_maker, tuple_of_types, args = (OXFORD_PATH_TXT,oxford_query_list,False, oxford_mean, oxford_std, IMG_SIZE))
else:
    raise ValueError('WHICH_DATASET has an invalid string.')
### ---------------------------------------------------------------------------

### Set up dataset for mAP calculations ---------------------------------------
if WHICH_DATASET == 'both':
    paris_labels_dictionary, paris_junk_dictionary = emb.images_with_labels(PARIS_PATH_TXT, paris_labels)
    paris_val_x = emb.validation_dataset_generator(PARIS_PATH, batchsize = VAL_BATCH_SIZE)
    oxford_labels_dictionary, oxford_junk_dictionary = emb.images_with_labels(OXFORD_PATH_TXT, oxford_labels)
    oxford_val_x = emb.validation_dataset_generator(OXFORD_PATH, batchsize = VAL_BATCH_SIZE)
else:
    raise ValueError('WHICH_DATASET has an invalid string.')
### ---------------------------------------------------------------------------

### Set up model --------------------------------------------------------------
if BASE_MODEL == 'VGG16':
    base_model = tf.keras.applications.VGG16(include_top=False,
                                             weights='imagenet',
                                             input_shape=IMG_SHAPE)
elif BASE_MODEL == 'ResNet50V2':
    base_model = tf.keras.applications.ResNet50V2(include_top=False,
                                             weights='imagenet',
                                             input_shape=IMG_SHAPE)
elif BASE_MODEL == 'MobileNetV2':
    base_model = tf.keras.applications.MobileNetV2(include_top=False,
                                             weights='imagenet',
                                             input_shape=IMG_SHAPE)
else:
    raise ValueError('BASE_MODEL has an invalid string.')


base_model.trainable = False

NetVLAD_layer = netvlad.NetVLAD(K=k_value)
dim_expansion = Lambda(lambda a: tf.expand_dims(a,axis=-2))
reduction_layer = Conv2D(embed_dimension,(1,1))
dense_reduction_layer = Dense(embed_dimension)
l2_normalization_layer = Lambda(lambda a: l2_normalize(a,axis=-1))

model = tf.keras.Sequential([
                             base_model,
                             NetVLAD_layer,
                             dense_reduction_layer,
                             l2_normalization_layer
                            ])
### ---------------------------------------------------------------------------

### Use pre-trained model or train conv5 too ----------------------------------
if USE_TRAINED_WEIGHTS:
    model = tf.keras.models.load_model(TRAINED_WEIGHTS_PATH, custom_objects={'NetVLAD': netvlad.NetVLAD})
else:
    pass

if TRAIN_CONV5:
    model.layers[0].trainable = True
    fine_tune_at = 15
    for layer in model.layers[0].layers[:fine_tune_at]:
        layer.trainable =  False
else:
    pass
### ---------------------------------------------------------------------------

### Training loop setup -------------------------------------------------------

# Create metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
valid_loss = tf.keras.metrics.Mean(name='valid_loss')
mean_ap = tf.keras.metrics.Mean(name='mean_ap')
mean_ap_semipos = tf.keras.metrics.Mean(name='mean_ap_semipos')

# Create optimizer
optimizer = tfa.optimizers.SGDW(weight_decay = 0.001, momentum = 0.9)

# Create LR scheduler
if VARIABLE_LR:
    lr_sched = tf.keras.experimental.CosineDecayRestarts(initial_learning_rate = LEARNING_RATE,
                                                         first_decay_steps = 10,
                                                         t_mul=1.0, m_mul=0.8, alpha=0.1)
else:
    lr_sched = tf.keras.experimental.CosineDecayRestarts(initial_learning_rate = LEARNING_RATE,
                                                         first_decay_steps = 1,
                                                         t_mul=1.0, m_mul=1.0, alpha=1.0)

# Create loss function (offline triplet mining)
def loss_func(embeds_q, embeds_p, embeds_n, margin=LOSS_MARGIN):
    
    d_pos = tf.reduce_sum(tf.math.square(embeds_q - embeds_p), 1)
    d_neg = tf.reduce_sum(tf.math.square(embeds_q - embeds_n), 1)

    loss = tf.maximum(0.0, margin + d_pos - d_neg)
    loss = tf.reduce_mean(loss)
    
    return loss

# Create train step
def train_step(batch):
    with tf.GradientTape() as tape:
        embeds_q = model(batch[0], training = True) # query, positive, negative
        embeds_p = model(batch[1], training = True)
        embeds_n = model(batch[2], training = True)
        loss = loss_func(embeds_q, embeds_p, embeds_n)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss.update_state(loss)
    
# Create validation step
def valid_step(batch):
    embeds_q = model(batch[0], training = False) # query, positive, negative
    embeds_p = model(batch[1], training = False)
    embeds_n = model(batch[2], training = False)
    loss = loss_func(embeds_q, embeds_p, embeds_n)
    
    valid_loss.update_state(loss)
    
# Create test step
def test_step(x, embed_path, path2txt, mean, std, label_list, labels_dictionary, junk_dictionary):
    print("Generating Embeddings...") # implement a progbar? but number of batches is unknown
    tick = time()
    emb.generate_all_embeddings(x, model, embed_path, VAL_BATCH_SIZE, mean, std, IMG_SIZE)
    tock = time()
    tf.print("Time taken to generate all embeddings: {}".format(tock-tick))
    count = 0
    print("Testing Against Queries...")
    for building in label_list:
        for query_number in range (1,6):
            ap = emb.test_metrics(embed_path, path2txt, building, query_number, labels_dictionary, junk_dictionary)
            mean_ap.update_state(ap)
            
            ap2 = emb.test_metrics_semipositive_junk(embed_path, path2txt, building, query_number, labels_dictionary, junk_dictionary)
            mean_ap_semipos.update_state(ap2)
            count += 1
            progbar2.update(count, values = [('mAP', mean_ap.result()),('mAP with Semipositives', mean_ap_semipos.result())])
            
# Create progbars
progbar = tf.keras.utils.Progbar(BATCH_SIZE, stateful_metrics = ['Loss'], verbose=1)
progbar_v = tf.keras.utils.Progbar(BATCH_SIZE, stateful_metrics = ['Validation Loss'], verbose=1)
progbar2 = tf.keras.utils.Progbar(55, stateful_metrics = ['mAP', 'mAP with Semipositives'])
### ---------------------------------------------------------------------------

### Training loop -------------------------------------------------------------
for epoch in range(EPOCHS):
    train_loss.reset_states()
    valid_loss.reset_states()
    
    paris_training_mapped = paris_training_x.map(pipe.decoder_function) # without running the decoder function again, the augmentations will be stale
    paris_validation_mapped = paris_validation_x.map(pipe.decoder_function)
    paris_train_ds = paris_training_mapped.batch(TRIPLET_SIZE)
    paris_valid_ds = paris_validation_mapped.batch(TRIPLET_SIZE)
    
    oxford_training_mapped = oxford_training_x.map(pipe.decoder_function) # without running the decoder function again, the augmentations will be stale
    oxford_validation_mapped = oxford_validation_x.map(pipe.decoder_function)
    oxford_train_ds = oxford_training_mapped.batch(TRIPLET_SIZE)
    oxford_valid_ds = oxford_validation_mapped.batch(TRIPLET_SIZE)

    for index, batch in enumerate(paris_train_ds):
        optimizer._set_hyper("learning_rate",lr_sched(epoch+STARTING_EPOCH))
        train_step(batch)
        progbar.update(index+1, values = [('Loss', train_loss.result())])
        if index+1 >= BATCH_SIZE:
            break
            
    for index, batch in enumerate(oxford_train_ds):
        optimizer._set_hyper("learning_rate",lr_sched(epoch+STARTING_EPOCH))
        train_step(batch)
        progbar.update(index+1, values = [('Loss', train_loss.result())])
        if index+1 >= BATCH_SIZE:
            break            
    
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch+STARTING_EPOCH)
        tf.summary.scalar('learning rate', optimizer._get_hyper("learning_rate"), step=epoch+STARTING_EPOCH)
        
    for index, batch in enumerate(paris_valid_ds):
        valid_step(batch)
        progbar_v.update(index+1, values = [('Validation Loss', valid_loss.result())])
        if index+1 >= BATCH_SIZE:
            break
            
    for index, batch in enumerate(oxford_valid_ds):
        valid_step(batch)
        progbar_v.update(index+1, values = [('Validation Loss', valid_loss.result())])
        if index+1 >= BATCH_SIZE:
            break            
            
    with validation_summary_writer.as_default():
        tf.summary.scalar('loss', valid_loss.result(), step=epoch+STARTING_EPOCH)   
        
    if (epoch == EPOCHS-1) or (epoch%20 == 0):
        mean_ap.reset_states()
        mean_ap_semipos.reset_states()
        test_step(paris_val_x, PARIS_EMBEDS, PARIS_PATH_TXT, paris_mean, paris_std, paris_labels, paris_labels_dictionary, paris_junk_dictionary)
        test_step(oxford_val_x, OXFORD_EMBEDS, OXFORD_PATH_TXT, oxford_mean, oxford_std, oxford_labels, oxford_labels_dictionary, oxford_junk_dictionary)
        with test_summary_writer.as_default():
            tf.summary.scalar('mAP', mean_ap.result(), step=epoch+STARTING_EPOCH)
            tf.summary.scalar('mAP with semi-positive', mean_ap_semipos.result(), step=epoch+STARTING_EPOCH)
        model.save(os.path.join(OUTPUT_WEIGHT_PATH,'{}_epo{}_model.h5'.format(current_time,epoch+STARTING_EPOCH)))

    template = 'Epoch {}, Loss: {}, Validation Loss: {}, mAP: {} (With semi-pos: {}) \n ------------'
    tf.print(template.format(epoch+STARTING_EPOCH + 1, train_loss.result(), valid_loss.result(), mean_ap.result(), mean_ap_semipos.result()))
### ---------------------------------------------------------------------------