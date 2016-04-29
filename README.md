# LRCN and Temporal CNN for Activity Recognition #

#### [Chih-Yao Ma](http://shallowdown.wix.com/chih-yao-ma), Min-Hung Chen (equal contribution)

## Abstract 
We examine and implement several leading techniques for Activity Recognition (video classification), while proposing and investigating a novel convolution on temporally-constructed feature vectors.

How we tackle Activity Recognition problem? 
##### CNN as baseline
<img src="/Figures/cnn.png" alt="CNN as baseline" height="200">
##### CNN + RNN [(LRCN)](http://jeffdonahue.com/lrcn/)
<img src="/Figures/lrcn.png" alt="CNN + RNN (LRCN)" height="200">
##### Temporal CNN
<img src="/Figures/tnn.png" alt="Temporal CNN)" height="200">


### Demo 
Demo video coming out soon!

### Dataset 
We are currently using [UCF101](http://crcv.ucf.edu/data/UCF101.php) dataset for our project. This dataset has 13320 videos from 101 action categories. 
<img src="http://crcv.ucf.edu/images/slideshow/UCF101.png" alt="UCF101 Dataset" height="200">

We will move onto [SPORTS-1M](http://cs.stanford.edu/people/karpathy/deepvideo/) dataset to see how much our performance will be changed in the near future. 
<img src="http://cs.stanford.edu/people/karpathy/deepvideo/sz70h.jpg" alt="SPORTS-1M Dataset" height="200">



## Installation 
Our work is currently implemented in Torch, and depends on the following packages: torch/torch7, torch/nn, torch/nngraph, torch/image, cudnn ...



## Acknowledgment 
This work was initialized as a class project for deep learning class in Georgia Tech 2016 Spring. We were teamed up with Hao Yan and Casey Battaglino to work on this class project, who have been a great help and provide valuable discussions as we go long this class project. 

#### This is an ongoing project. Please contact us if you have any questions. 
[Chih-Yao Ma](http://shallowdown.wix.com/chih-yao-ma) at <cyma@gatech.edu>

Min-Hung Chen at <cmhungsteve@gatech.edu>


