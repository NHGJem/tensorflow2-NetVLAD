import numpy as np
import pandas as pd 
import os

import tensorflow as tf

import data_pipeline as pipe

# =============================================================================
#
# This script contains functions for generating and testing embeddings to get mAP.
# It will ignore images marked as junk by the groundtruth txt files, but this can be changed in test_metrics()
#
# =============================================================================

# %% [code]
blacklist = ["paris_louvre_000136.jpg",
        "paris_louvre_000146.jpg",
        "paris_moulinrouge_000422.jpg",
        "paris_museedorsay_001059.jpg",
        "paris_notredame_000188.jpg",
        "paris_pantheon_000284.jpg",
        "paris_pantheon_000960.jpg",
        "paris_pantheon_000974.jpg",
        "paris_pompidou_000195.jpg",
        "paris_pompidou_000196.jpg",
        "paris_pompidou_000201.jpg",
        "paris_pompidou_000467.jpg",
        "paris_pompidou_000640.jpg",
        "paris_sacrecoeur_000299.jpg",
        "paris_sacrecoeur_000330.jpg",
        "paris_sacrecoeur_000353.jpg",
        "paris_triomphe_000662.jpg",
        "paris_triomphe_000833.jpg",
        "paris_triomphe_000863.jpg",
        "paris_triomphe_000867.jpg",]


def images_with_labels(path2txt, label_list, filetype='jpg'):
    path2txt = path2txt#.decode()
    
    label_dict = {}
    junk_dict = {}
    
    for i in range(len(label_list)):
        label = label_list[i]#.decode()
        
        for j in range(1,6):
            query_name = label+'_{}'.format(j)
            
            query = query_name+'_query.txt'
            good = query_name+'_good.txt'
            ok = query_name+'_ok.txt'
            junk = query_name+'_junk.txt'
            
            ### get the query image path from the _query.txt
            with open(os.path.join(path2txt,query)) as file:
                contents = file.read().split(' ') 
                query_image_name = contents[0].replace('oxc1_','')+"."+filetype #remove the oxc1_ part which exists in the oxford query txt
                
            list_of_good = pipe.generate_img_list(path2txt, good)
            list_of_ok = pipe.generate_img_list(path2txt, ok)
            list_of_junk = pipe.generate_img_list(path2txt, junk)
            
            tmp_list = list_of_good + list_of_ok
            tmp_list = list(set(tmp_list))
        
            label_dict[query_name] = tmp_list
            
            tmp_list2 = list_of_junk
            tmp_list2 = list(set(tmp_list2))
            
            junk_dict[query_name] = tmp_list2
        
    return label_dict, junk_dict

def validation_dataset_generator(pathname, batchsize = 64):
    list_of_names = []
    for dirname, _, list_of_filenames in os.walk(pathname):
        for filename in list_of_filenames:
            if filename in blacklist:
                pass
            else:
                img_path = os.path.join(dirname, filename)
                list_of_names.append((img_path, filename.rstrip('.jpg')))
        ds = tf.data.Dataset.from_tensor_slices(list_of_names)
        ds = ds.batch(batchsize)
        return ds


def embedding_generator(paths,model,batchsize, mean, std):    
    tensors = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for image in paths:
        decoded = pipe.decode_only(image, mean, std)
        tensors = tensors.write(tensors.size(),decoded)
    tensors = tensors.stack()
    embeddings = model(tensors, training = False)
    return embeddings

def make_one_npy(npy_name,tensor,embed_folder):
    name = np.array2string(np.asarray(npy_name))      # b'paris_general_002035' -- currently breaks here if run in graph exec
    name = name.replace('b\'','').replace('\'','')    # paris_general_002035
    np.save(os.path.join(embed_folder,name),tensor)
    return

def generate_all_embeddings(ds,model,embed_path,batchsize, mean, std):
    for batch in ds: # (32, 2)
        paths = batch[:,0]                         # (32, ) containing paths of the 32 images
        embeds = embedding_generator(paths,model,batchsize, mean, std)  # outputs embeddings of the 32
        embeds = tf.unstack(embeds)                # convert into an (iteratable) list of 32 embeddings 
        label = batch[:,1]                         # (32, ) containing names of the 32 images
        for index,one_embed in enumerate(embeds):
            make_one_npy(label[index],one_embed,embed_path)
    return

def test_metrics(embed_path, path2txt, building, query_number, labels_dictionary, junk_dictionary, k):
    
    query_name = building+'_{}'.format(query_number)

    query = query_name+'_query.txt'

    ### get the query image path from the _query.txt
    with open(os.path.join(path2txt,query)) as file:
        contents = file.read().split(' ') 
        query_image_name = contents[0].replace('oxc1_','')+".npy" #remove the oxc1_ part which exists in the oxford query txt

    query_embed = np.load(os.path.join(embed_path,query_image_name))

    # Create similarity list
    similarity_list = []
    name_list = []
    for dirname, _, list_of_filenames in os.walk(embed_path):
        for filename in list_of_filenames:
            if filename != query_image_name:
                compare_embed = np.load(os.path.join(embed_path,filename))
                euc_dist = np.sum(np.square(query_embed - compare_embed))
                similarity_list.append(euc_dist)
                name_list.append(filename)
    similarity_list = np.asarray(similarity_list)
    indexes = (similarity_list).argsort()[:k]
    best_matches = [name_list[index] for index in indexes]
    
    # example: [1,1,0,0,1,1], 5 total positives: (1+1+0+0+3/5+4/6)/5
    total_num_positives = len(labels_dictionary[query_name])
    numerator = 0
    denominator = 0
    sum_of_fractions = 0
    for img_name in best_matches:
        img_name = img_name.replace('.npy','.jpg')
        if img_name in labels_dictionary[query_name]:
            numerator += 1
            denominator += 1
            sum_of_fractions += (numerator/denominator)
        elif img_name in junk_dictionary[query_name]:
            #denominator += 1  # if this is commented out, junk images are ignored. otherwise, junk images are negative. copy code from the if-block to make junk images positive.
            pass 
        else:
            denominator += 1

    if total_num_positives > denominator:
        ap = sum_of_fractions/denominator
    else:
        ap = sum_of_fractions/total_num_positives

    recall = numerator/total_num_positives

    return ap, recall