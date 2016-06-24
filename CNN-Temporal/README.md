# Temporal Convolutional Neural Network
This folder contains codes to generate feature vectors from optical flow images of the UCF-101 database. The generated features are used for our network for temporal design: baseline CNN, CNN+RNN, and TCNN.

1. Load most of the videos in UCF-101 (use CUDA for default).
2. Use the model "ResNet-101".
3. Extract the feature before the full-connected layer.
4. Generate the name list as well.
5. No need to specify the video number ==> use all the videos above the user-defined minimal frame#.
6. Follow the original author's split sets to generate 3 training/testing sets.


---
## Requirement
### Dataset
We will provide the generated optical flow videos for UCF-101 soon.

### Trained model
The CNN model trained on optical flow images were initially pre-trained on ImageNet.
We will provide the pre-trained model for generating features from optical flow images soon.

### Libraries
There are some libraries and you need to install before running.

#### ffmpeg library:
```bash
$ sudo apt-get install ffmpeg
$ luarocks install ffmpeg
```
To solve the installation problems for for Ubuntu 14.04:
```bash
$ sudo add-apt-repository ppa:mc3man/trusty-media
$ sudo apt-get update
$ sudo apt-get dist-upgrade
$ sudo apt-get install ffmpeg
```

### Other files
transforms.lua: pre-process images (scaling, normalization and cropping)

---
## Usage
```bash
$ qlua temporalNetwork.lua
```
Tunable parameters (around line 83):
* numFrameMin:	frame # you want to extract for each video (default: 50)
* numSplit:			training/testing split # of UCF-101 (default: 1)

### Some parameters:
* class# = 101
* feature dimension = 2048

### Outputs
* name: 		video name
* path:		local video path (under "UCF-101/")
* featMats: 	total video# x feature dimension x frame#
* labels:		total video# x 1

---
## Other notes:
After running these codes, a new empty folder "out_frames" will be generated. You can ignore it. That's only for debugging.

---
#### Contact
[Chih-Yao Ma](http://shallowdown.wix.com/chih-yao-ma) at <cyma@gatech.edu> or [[LinkedIn]](https://www.linkedin.com/in/chih-yao-ma-9b5b3063)

[Min-Hung Chen](https://www.linkedin.com/in/chensteven) at <cmhungsteve@gatech.edu>
