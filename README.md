# tensorflow2-NetVLAD
NetVLAD in Tensorflow 2.0 for training on Oxford5k/Paris6k using triplet loss.

# Data
[Paris](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) and [Oxford](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) datsets. ROI not used.

# Setup and Usage
Change the variables in main.py, ensure all paths to the image folder and ground truths are set correctly, then run main.py.

# Mean Average Precision Results
Source | Paris | Oxford
--- | --- |---
**NetVLAD Paper - Trained on Pittsburgh** | 78.5% | 69.1%
**Ours - Training images unaugmented** | 87.4% | 79.6%
**Ours - Training images augmented** | 85.7% | 72.1%

