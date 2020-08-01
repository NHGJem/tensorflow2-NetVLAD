import numpy as np
import pandas as pd 
import os
import itertools
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_addons as tfa

from numpy.random import default_rng
rng = default_rng()

# =============================================================================
#
# This script contains functions for generating training and validation datasets.
#
# =============================================================================


def generate_query_list(path2txt, filetype='jpg'):
    query_txt = []

    for dirname, _, filenames in os.walk(path2txt):
        for filename in filenames:
            if 'query' in filename:
                query_txt.append(filename)
    
    return query_txt


def generate_img_list(path2txt, filename, filetype='jpg'):
    img_list = []
    
    with open(os.path.join(path2txt,filename)) as file:
        for line in file:
            img_list.append(line.replace('\n','')+"."+filetype)
    
    return img_list


def triplet_generator(path2txt, querytxt, train, mean, std, filetype='jpg'):
    path2txt = path2txt.decode()
    
    list_of_paths = []
    
    ### get a random query
    query = querytxt[rng.integers(len(querytxt))].decode()

    ### confirm the name of the query in the form of "<building>_<number>"
    triplet_name = query.rstrip('_query.txt')

    ### get the query image in the form of its .jpg under attribute query_image_name
    with open(os.path.join(path2txt,query)) as file:
        contents = file.read().split(' ') 
        query_image_name = contents[0].replace('oxc1_','')+"."+filetype #remove the oxc1_ part which exists in the oxford query txt
    query_image_path = os.path.join(os.path.dirname(path2txt), query_image_name)

    ### get the other txt files related to the query
    good = triplet_name+'_good.txt'
    ok = triplet_name+'_ok.txt'
    junk = triplet_name+'junk.txt'    

    list_of_good = generate_img_list(path2txt, good)
    list_of_ok = generate_img_list(path2txt, ok)
    
    ### generate a list of positives from good and ok and pick one under attribute positive_image_name
    list_of_positives = []

    list_of_good = generate_img_list(path2txt, good)
    list_of_ok = generate_img_list(path2txt, ok)

    list_of_positives = list_of_good+list_of_ok

    ### Get a small handful of images from only the first 80%. Comment out the split and if-else if training on all images.
    split = int(tf.math.ceil(len(list_of_positives)*0.8))
    if train:
        list_of_positives = list_of_positives[:split]
    else:
        list_of_positives = list_of_positives[split:]

    positive_image_name = list_of_positives[rng.integers(len(list_of_positives))]
    positive_image_path = os.path.join(os.path.dirname(path2txt), positive_image_name)

    ### generate a negative by taking another query, ensuring a different building, and pick one of its positives under attribute negative_image_name
    building = triplet_name.rstrip('_12345')
    wrong_triplet_name = querytxt[rng.integers(len(querytxt))].decode().rstrip('_query.txt')
    wrong_building = wrong_triplet_name.rstrip('_12345')

    while wrong_building == building:
        wrong_triplet_name = querytxt[rng.integers(len(querytxt))].decode().rstrip('_query.txt')
        wrong_building = wrong_triplet_name.rstrip('_12345')

    wrong_good = wrong_triplet_name+'_good.txt'
    wrong_ok = wrong_triplet_name+'_ok.txt'

    list_of_negatives = generate_img_list(path2txt, wrong_good) + generate_img_list(path2txt, wrong_ok)

    ### Get a small handful of images from only the first 80%. Comment out the split and if-else if training on all images.
    split = int(tf.math.ceil(len(list_of_negatives)*0.8))
    if train:
        list_of_negatives = list_of_negatives[:split]
    else:
        list_of_negatives = list_of_negatives[split:]

    negative_image_name = list_of_negatives[rng.integers(len(list_of_negatives))]
    negative_image_path = os.path.join(os.path.dirname(path2txt), negative_image_name)

    list_of_paths.append(query_image_path)
    list_of_paths.append(positive_image_path)
    list_of_paths.append(negative_image_path)
        
    return tuple(list_of_paths), train, mean, std


def dataset_maker(path2txt, querytxt, train, mean, std):
    for i in itertools.count():
        yield triplet_generator(path2txt, querytxt, train, mean, std)

'''def decode_augment_old(path, mean, std, img_size = 224):
    
    max_dim = tf.convert_to_tensor([img_size,img_size], dtype=tf.int32)
    
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    img = tf.image.resize(img, max_dim)
    
    img = tf.math.subtract(img,mean)
    img = tf.math.divide(img,std)
    
    ## Cropping zone
    roll = tf.random.uniform([])
    fraction = rng.uniform(low=0.5)
    if roll < 0.3:
        img = tf.image.central_crop(img,fraction)
    elif roll < 0.6:
        crop_dim = tf.math.multiply(tf.cast(max_dim, tf.float32),fraction)
        crop_dim = tf.cast(tf.math.round(crop_dim), tf.int32)
        crop_dim = tf.concat([crop_dim, [3]], 0)
        img = tf.image.random_crop(img,crop_dim)
    else:
        pass        
    img = tf.image.resize(img, max_dim)

    ## Color ops zone
    if tf.random.uniform([]) < 0.3:
        img = tf.image.random_brightness(img, 0.2)
    if tf.random.uniform([]) < 0.3:
        img = tf.image.random_contrast(img, 1, 2)
    if tf.random.uniform([]) < 0.3:
        img = tf.image.random_saturation(img, 1, 3)

    ## Transform ops zone
    radian = rng.uniform(low=-0.2, high=0.2)
    img = tfa.image.rotate(img, radian)

    roll = rng.uniform()
    if roll < 0.3:
        roll = round(roll*img_size)
        dx = rng.uniform(-roll,roll)
        dy = rng.uniform(-roll,roll)
        img = tfa.image.translate(img, [dx,dy])

    return img'''

def decode_augment(path, mean, std, img_size = 224):
    
    max_dim = tf.convert_to_tensor([img_size,img_size], dtype=tf.int32)
    
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    if h > w:
        h = w
    else:
        w = h
    
    img = tf.image.resize_with_crop_or_pad(img, h, w)
    img = tf.image.resize(img, max_dim)
    
    ## Photometric Augmentations - Apply brightness, or contrast, or saturation, or none
    
    roll = rng.uniform()
    
    if roll < 0.25:
        img = tf.image.random_brightness(img, 0.2)
    elif roll < 0.50:
        img = tf.image.random_contrast(img, 1, 2)
    elif roll < 0.75:
        img = tf.image.random_saturation(img, 1, 3)
    else:
        pass
    
    ## Apply dataset normalization
    
    img = tf.math.subtract(img,mean)
    img = tf.math.divide(img,std)
    
    ## Geometric Augmentations - Apply translate, or central crop, or random crop, or rotate, or horizontal flip, or none
    
    roll = rng.uniform()
    fraction = rng.uniform(low=0.7)
    
    if roll < 0.2:
        delta = round(roll/4*img_size)
        dx = rng.uniform(-delta,delta)
        dy = rng.uniform(-delta,delta)
        img = tfa.image.translate(img, [dx,dy])
    elif roll < 0.3:
        img = tf.image.central_crop(img,fraction)    
    elif roll < 0.4:
        crop_dim = tf.math.multiply(tf.cast(max_dim, tf.float32),fraction)
        crop_dim = tf.cast(tf.math.round(crop_dim), tf.int32)
        crop_dim = tf.concat([crop_dim, [3]], 0)
        img = tf.image.random_crop(img,crop_dim)
    elif roll < 0.6:
        radian = rng.uniform(low=-0.2, high=0.2)
        img = tfa.image.rotate(img, radian)
    elif roll < 0.8:
        img = tf.image.flip_left_right(img)
    else:
        pass
    
    img = tf.image.resize(img, max_dim)    

    return img

def decode_only(path, mean, std, img_size = 224):
    
    max_dim = tf.convert_to_tensor([img_size,img_size], dtype=tf.int32)
    
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    if h > w:
        h = w
    else:
        w = h
    
    img = tf.image.resize_with_crop_or_pad(img, h, w)
    img = tf.image.resize(img, max_dim)
    
    img = tf.math.subtract(img,mean)
    img = tf.math.divide(img,std)

    return img

def decoder_function(paths, train, mean, std, img_size = 224):
    
    single_input = []
    mean = tf.broadcast_to(mean,[224,224,3])
    std = tf.broadcast_to(std,[224,224,3])
    
    for image in paths:
        decoded = tf.cond(train == True, true_fn = lambda: decode_augment(image, mean, std, img_size), false_fn = lambda: decode_only(image, mean, std, img_size))
        single_input.append(decoded)
    
    return single_input

# For plotting images to check them
def show_batch(ds):
    titles = ['query','positive','negative']
    plt.figure(figsize=(20,20))

    for batch in ds:
        
        query = batch[0]
        n = 1
        for image in query:
            ax = plt.subplot(33,3,n)
            n += 3
            ax.set_title(titles[0])
            ax.imshow(image)
            plt.axis('off')
            
        positive = batch[1]
        n = 2
        for image in positive:
            ax = plt.subplot(33,3,n)
            n += 3
            ax.set_title(titles[1])
            ax.imshow(image)
            plt.axis('off')
            
        negative = batch[2]
        n = 3
        for image in negative:
            ax = plt.subplot(33,3,n)
            n += 3
            ax.set_title(titles[2])
            ax.imshow(image)
            plt.axis('off')
        
        break