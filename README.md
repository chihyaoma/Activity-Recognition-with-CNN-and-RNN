# Activity Recognition with RNN and Temporal-ConvNet
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Chih-Yao Ma](http://shallowdown.wixsite.com/chih-yao-ma/me)\*, [Min-Hung Chen](https://www.linkedin.com/in/chensteven)\*

(\* equal contribution)




### This project will soon be largely revised and updated. Please stay tuned.



---
## Abstract

We examine and implement several leading deep learning techniques for Human Activity Recognition (video classification), while proposing and investigating a novel convolution on temporally-constructed feature vectors.

Our proposed model classify videos into different human activities and give confident scores for each prediction. Features extracted from both spatial and temporal network were integrated by RNN to make prediction for each image frame. Class predictions for each of the video are made by voting through several selected video frames.


#### How we tackle Activity Recognition problem?
CNN as baseline, CNN + RNN [(LRCN)](http://jeffdonahue.com/lrcn/), Temporal CNN

<table align = "center">
<tr>
  <td align = "center"> CNN as baseline </td>
  <td align = "center"> CNN + RNN (LRCN)</td>
  <td align = "center"> Temporal CNN </td>
</tr>
<tr>
<td> <img src="https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN/blob/master/figures/cnn.png?raw=true" alt="CNN as baseline" height="120"></td>
<td> <img src="https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN/blob/master/figures/lrcn.png?raw=true" alt="CNN + RNN (LRCN)" height="120"></td>
<td> <img src="https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN/blob/master/figures/tnn.png?raw=true" alt="Temporal CNN" height="120"> </td>
</tr>
</table>

<!-- <img src="/Figures/cnn.png" alt="CNN as baseline" height="200">
##### CNN + RNN [(LRCN)](http://jeffdonahue.com/lrcn/)
<img src="/Figures/lrcn.png" alt="CNN + RNN (LRCN)" height="200">
##### Temporal CNN
<img src="/Figures/tnn.png" alt="Temporal CNN)" height="200"> -->


### Demo

<p align="center">
<a href="http://www.youtube.com/watch?feature=player_embedded&v=81FSYgw6BVA" target="blank"><img src="http://img.youtube.com/vi/81FSYgw6BVA/0.jpg"
alt="IMAGE ALT TEXT HERE" width="360" height="270" border="10" /></a>
</p>

The above YouTube video demonstrates the top-3 predictions results of our LRCN and temporal CNN model. The text on the top is the ground truth, three texts are the predictions for each of the method, and the bar right next to the predictions are how confident the model makes predictions.


<p align="center">
<img src="https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN/blob/master/figures/demo.gif?raw=true" width="250">
<img src="https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN/blob/master/figures/demo-1.gif?raw=true" width="250">
<img src="https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN/blob/master/figures/demo-2.gif?raw=true" width="250">
</p>
---
## Dataset
We are currently using [UCF101](http://crcv.ucf.edu/data/UCF101.php) and [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) dataset for our project.


---
## Installation
Our work is currently implemented in Torch, and depends on the following packages: torch/torch7, torch/nn, torch/nngraph, torch/image, cudnn ...

If you are on Ubuntu, please follow the instruction here to install Torch. For a more comprehensive installation guilde, please check [Torch installation](http://torch.ch/docs/getting-started.html) or [Self-contained Torch installation](https://github.com/torch/distro).

```bash
$ git clone https://github.com/torch/distro.git ~/torch --recursive
$ cd ~/torch; bash install-deps;
$ ./install.sh
$ source ~/.bashrc

```
You will also need to install some of the packages we used from LuaRocks. LuaRocks should already be installed with your Torch.
```bash
$ luarocks install torch
$ luarocks install pl
$ luarocks install trepl
$ luarocks install image
$ luarocks install nn
$ luarocks install dok
$ luarocks install gnuplot
$ luarocks install qtlua
$ luarocks install sys
$ luarocks install xlua
$ luarocks install optim
```
If you would like to use CUDA on your NVIDIA graphic card, you will need to install [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) and some additional packages.
For installing CUDA 8.0 on Ubuntu:
```bash
# cd to where the downloaded file located, and then
$ sudo dpkg -i cuda-repo-ubuntu1404-8-0-local_8.0.44-1_amd64.deb
$ sudo apt-get update
# install cuda using apt-get
$ sudo apt-get install cuda
```
add the following lines to your ~/.bashrc file
```bash
export CUDA_HOME=/usr/local/cuda-8.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64

PATH=${CUDA_HOME}/bin:${PATH}
export PATH
```
Remember to source your bashrc file afterwards.
```bash
$ source ~/.bashrc
```
In order to use CUDA with Torch, you will need to install some additional packages.
```bash
$ luarocks install cutorch
$ luarocks install cunn
```
You **need to** install the CUDNN package properly since we use the pre-trained ResNet model. First, you need to download the package from [Nvidia](https://developer.nvidia.com/cudnn) (You need to register to download it.)

Then, follow this instruction:
```bash
# cd to where the downloaded file located, and then
$ tar -zxf cudnn-8.0-linux-x64-v5.1.tgz
$ cd cuda
$ sudo cp -a lib64/* /usr/local/cuda/lib64/
$ sudo cp include/* /usr/local/cuda/include/
$ luarocks install cudnn
```

---
## Usage
We provide three different methods to train the models for activity recognition: CNN, CNN with RNN, and Temporal CNN.

#### Inputs
Our models will take the **feature vectors** generated by the first CNN as input for training. You can generate the features using our codes under "/CNN_Spatial/". You can also download the feature vectors generated by ourselves. (please refer to the Dropbox link below.) We followed the first training/testing split from [UCF-101](http://crcv.ucf.edu/data/UCF101.php). If you would like to compare with our results, please use the same training and testing list, as it will affect your overall performance a lot.

* [Features for training](https://www.dropbox.com/s/ehla4szd8z8u8lw/data_UCF101_train_1.t7?dl=0)
* [Features for testing](https://www.dropbox.com/s/cma4swez0fabw47/data_UCF101_test_1.t7?dl=0)

#### CNN with RNN
We use the [RNN library](https://github.com/Element-Research/rnn) provided by Element-Research. Simply install it by:
```bash
$ luarocks install rnn
```
After you downloaded the feature vectors, please modify the code in *./RNN/data.lua* to the director where you put your feature vector files.

To start the training process, go to *./RNN* and simply execute:
```bash
$ th main.lua
```
The training and testing loss will be reported, and the results will be saved into log files. The learning rate and best testing accuracy will be reported each epoch if there is any update.

#### Temporal-ConvNet
To start the training process, go to *./TCNN* and simply execute:
```bash
$ qlua run.lua -r 15e-5
```
For more details, please refer to the readme file in the folder *./TCNN/*.

You also need to modify the code in *./TCNN/data.lua* to the director where you put your feature vector files.

The training and testing performance will be plotted, and the results will be saved into log files. The best testing accuracy will be reported each epoch if there is any update.

---
## Acknowledgment
This work was initialized as a class project for deep learning class in Georgia Tech 2016 Spring. We were teamed up with Hao Yan and Casey Battaglino to work on this class project, who have been a great help and provide valuable discussions as we go long this class project.

#### This is an ongoing project. Please contact us if you have any questions.

[Chih-Yao Ma](http://shallowdown.wix.com/chih-yao-ma/me) at <cyma@gatech.edu> or [[LinkedIn]](https://www.linkedin.com/in/chih-yao-ma-9b5b3063)

[Min-Hung Chen](https://www.linkedin.com/in/chensteven) at <cmhungsteve@gatech.edu>
