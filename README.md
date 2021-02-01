# KD-Ticket

Code for [Good Students Play Big Lottery Better](https://arxiv.org/abs/2101.03255)


## Overview
* We reveal a new opportunity for finding lottery tickets in large-scale deep networks, by introducing the **KD ticket** to recycle the soft-labels from dense networks, as an extra modellevel cue to guide sparse network training.
* We demonstrate that our KD ticket can be compatible with other "rewinding" methods, dubbed **KD-rewinding**, and in this way further performance boost can be achieved.


## Prepare

````
Pytorch >= 1.4.0  
torchvision >= 0.5.0  
cuda >= 10.1   
progress >= 1.5  
````

If you wish to train with 16-bit precision on ImageNet, you need to install the [NVIDIA Apex](https://anaconda.org/conda-forge/nvidia-apex) 

For ImageNet, you should ask for permission and download it from their [website](http://www.image-net.org/) 


##  Experiments
For experiments, please go to README.md in /CIFAR10 and /ImageNet 


## Reference
[Rethinking the Value of Network Pruning](https://github.com/Eric-mingjie/rethinking-network-pruning)    
[open-lth](https://github.com/facebookresearch/open_lth)


