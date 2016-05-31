# Spatial Convolutional Neural Network
This folder contains codes to generate feature vectors of the UCF-101 database. The generated features are used for our network for temporal design: baseline CNN, CNN+RNN, and TCNN.

1. Load most of the videos in UCF-101 (use CUDA for default).
2. Use the model "ResNet-101".
3. Extract the feature before the full-connected layer.
4. Generate the name list as well.
5. No need to specify the video number ==> use all the videos above the user-defined minimal frame#.
6. Follow the original author's split sets to generate 3 training/testing sets.


---
## Requirement
### Dataset
There are more details for downloading the [UCF-101](http://crcv.ucf.edu/data/UCF101.php) dataset.


You can ownload from the website or use the command line:
```bash
$ wget http://crcv.ucf.edu/data/UCF11_updated_mpg.rar
```
if you dont have unrar (or use sudo apt-get):
```bash
$ wget http://www.rarlab.com/rar/rarlinux-3.6.0.tar.gz
$ tar -zxvf rarlinux-*.tar.gz
$ ./rar/unrar x UCF11_updated_mpg.rar
```
To solve the problems for running the 32-bit unrar on a 64-bit system.
* For Ubuntu 13.10 and above
```bash
$ sudo dpkg --add-architecture i386
$ sudo apt-get update
$ sudo apt-get install libncurses5:i386 libstdc++6:i386 zlib1g:i386
```
* For earlier version
```bash
$ apt-get install ia32-libs
```

### Pre-trained model
We used a pre-trained CNN model [Res-101](http://torch7.s3-website-us-east-1.amazonaws.com/data/resnet-101.t7) to generate the features.

You may need to change the directory paths of the model and dataset in the code (around line 78).
```bash
dirModel = './models/'
dirDatabase = '/home/cmhung/Desktop/UCF-101/'
```

### Libraries
There are some libraries and you need to install before running my codes.

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
$ th run_UCF101_final_ResNet.lua
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
[Min-Hung Chen](https://www.linkedin.com/in/chensteven) at <cmhungsteve@gatech.edu>

Last updated: 05/05/2016
