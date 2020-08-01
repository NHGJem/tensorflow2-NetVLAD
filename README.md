# tensorflow2-NetVLAD
NetVLAD in Tensorflow 2.0 for Deep Image Retrieval. Trained on Oxford5k/Paris6k using triplet loss.

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Loss function](#loss-function)
- [Deep Learning Architecture](#deep-learning-architecture)
- [Metrics](#metrics)
- [Hyperparameters](#hyperparameters)
- [Data Augmentations](#data-augmentations)
- [Setup and Usage](#setup-and-usage)
- [Results](#results)
   - [Paris Dataset](#paris-dataset)
   - [Oxford Dataset](#oxford-dataset)
- [Result Graphs](#result-graphs)
   - [Paris Dataset](#training-and-validation-loss-curve-of-paris)
   - [Oxford Dataset](#training-and-validation-loss-curve-of-oxford)
   
## Introduction
Goal of this project is to construct a NetVLAD model in Tensorflow 2 for end-to-end learning of deep image retrieval. Images are learnt as embeddings and then ranked according to shortest euclidean distance. 

## Data
The popular [Paris](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) and [Oxford](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) datsets for image retrieval were used. The provided ROI bounding boxes were not used.

Data was split into 80:20 for training and validation.

## Loss Function
The loss function used is triplet loss with a margin, where the triplets are generated by simple offline mining.

The negative image used for any query and positive image was a positive image of a different building, as such buildings would tend to have similar-looking features that we would want our model to be able to differentiate.

## Deep Learning Architecture
The NetVLAD layer, originally designed usign MatLab ([Github Page](https://github.com/Relja/netvlad)), was modified to be a custom keras layer compatible with Tensorflow 2 ([Github Page](https://github.com/crlz182/Netvlad-Keras)). The layer is then joined to the VGG16 model with its Fully-Connected layers removed, and weights pre-trained on ImageNet.

![Image of Model](https://github.com/NHGJem/tensorflow2-NetVLAD/blob/master/readme_images/model.png)

## Metrics
Mean Average Precision (mAP) over all querys were used, where images labelled "ok" and "good" were treated as positives and images labelled "junk" were entirely removed from the calculation.

An alternative method of calculating mAP was also tested, that considered junk images as "semi-positives".         
If a junk image was ranked to be closer in similarity to the query compared to positives, it would contribute towards the mAP but at a reduced value. This value is low if the junk image was ranked to be closer than many positives (effectively treating it like a negative), but increases as more positives are ranked before the junk image, until all positives have been ranked.                 
After this threshold has been reached, all remaining junk images would be considered full positives, with the intention that the model should rank these junk images closer than negative images.

A model swap was also performed, where the model trained on Paris was used to evaluate Oxford and vice-versa. This is to investigate the generalizability of the model.

## Hyperparameters
Object | Value
 --- | --- 
 Image size | (224, 224, 3) 
 Batch size | 20 (Parameters updated every 16 triplets)
 Epoch | 200 (Maximum mAP tends to be achieved before 200) 
 Loss Margin | 0.1
 Optimizer | Stochastic Gradient Descent 
 Learning Rate Scheduler | Cosine Decay with Restarts 
 Initial Learning Rate | 1.0e-3 
 Minimum Learning Rate | 1.0e-4 
 Length of each decay period | 10 epochs 
 Subsequent initial learning rates | 0.8x of previous period's initial 
 
 ![Chart of LR for each epoch](https://github.com/NHGJem/tensorflow2-NetVLAD/blob/master/readme_images/lr_chart.png)

## Data Augmentations
Standard augmentations were used. These include: 
- Photometric augmentations like brightness, contrast, and saturation
- Geometric augmentations like zoom, rotation, and translation

Prior to augmentation, non-square images were cropped into squares to preserve aspect ratio.

Tests were also performed without augmentations, for comparison.

## Setup and Usage
Change the variables in main.py, ensure all paths to the image folder and ground truths are set correctly, then run main.py.

## Results

### **Paris Dataset**
mAP | Ignore Junk | Semipositive Junk
--- | --- |---
**NetVLAD Paper - Trained on Pittsburgh** | 78.5% | -
**Ours - Unaugmented** | 87.4% | 60.4%
**Ours - Augmented** | 88.2% | 57.9%
**Ours - Model trained on Augmented Oxford dataset** | 49.4% | 39.7%

### **Oxford Dataset**
mAP | Ignore Junk | Semipositive Junk
--- | --- |---
**NetVLAD Paper - Trained on Pittsburgh** | 69.1% | -
**Ours - Unaugmented** | 79.6% | 60.5%
**Ours - Augmented** | 76.5% | 49.7%
**Ours - Model trained on Augmented Paris dataset** | 37.0% | 32.5%

## Result Graphs

### **Training and Validation Loss Curve of Paris**
Due to interruptions in training, it had to be restarted from checkpoints. As such, the complete curves are made from the curves of three logs.

![Paris Training Loss](https://github.com/NHGJem/tensorflow2-NetVLAD/blob/master/readme_images/paris_trainloss.png)
![Paris Validation Loss](https://github.com/NHGJem/tensorflow2-NetVLAD/blob/master/readme_images/paris_validloss.png)

### **mAP of model (Augmented Paris Images)**
![Paris mAP](https://github.com/NHGJem/tensorflow2-NetVLAD/blob/master/readme_images/paris_map.png)

### **Training and Validation Loss Curve of Oxford**
![Oxford Training Loss](https://github.com/NHGJem/tensorflow2-NetVLAD/blob/master/readme_images/oxford_trainloss.png)
![Oxford Validation Loss](https://github.com/NHGJem/tensorflow2-NetVLAD/blob/master/readme_images/oxford_validloss.png)

### **mAP of model (Augmented Oxford Images)***
![Oxford mAP](https://github.com/NHGJem/tensorflow2-NetVLAD/blob/master/readme_images/oxford_map.png)
