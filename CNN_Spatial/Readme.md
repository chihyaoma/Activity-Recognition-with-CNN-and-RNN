# Spatial Convolutional Neural Network
This folder contains codes to generate feature vectors of the UCF-101 database. The generated features are used for our network for temporal design: baseline CNN, CNN+RNN, and TCNN

---
## Dataset
There are more details for downloading the [UCF101](http://crcv.ucf.edu/data/UCF101.php) dataset.


You can ownload from the website or use the command line:
```bash
wget http://crcv.ucf.edu/data/UCF11_updated_mpg.rar
```
if you dont have unrar (or use sudo apt-get):
```bash
wget http://www.rarlab.com/rar/rarlinux-3.6.0.tar.gz
tar -zxvf rarlinux-*.tar.gz
./rar/unrar x UCF11_updated_mpg.rar
```

## Requirement
There are some libraries and you need to install before running my codes.



### get ffmpeg library:
sudo apt-get install ffmpeg

### get video libraries for lua:
luarocks install ffmpeg

### Common Problems
1. ffmpeg installation problems for Ubuntu 14.04:

	sudo add-apt-repository ppa:mc3man/trusty-media

	sudo apt-get update

	sudo apt-get dist-upgrade

	sudo apt-get install ffmpeg

2. run 32-bit unrar in a 64-bit system
	* For Ubuntu 13.10 and above

		sudo dpkg --add-architecture i386

		sudo apt-get update

		sudo apt-get install libncurses5:i386 libstdc++6:i386 zlib1g:i386

	* For earlier version

		apt-get install ia32-libs

---
## File List & Implementation details & Usage:
If you don't want to try all the codes, you can just try the first one, which is used to generate our final Res-101 features.

-----------------------------------------------------------------------------
### run_UCF101_final_ResNet.lua:
1. Load most of the videos in UCF-101 (use CUDA for default)
2. Use the model "ResNet-101"
3. extract the feature before the full-connected layer
4. generate the name list as well
5. No need to specify the video number ==> use all the videos above the user-defined minimal frame#
6. Follow the original author's split sets to generate 3 training/testing sets

command: th run_UCF101_final_ResNet.lua

notes:

1. You need to download the dataset and modify the path in the code
2. parameters:
	* class# = 101
	* feature dimension = 2048
	* frame# = 50 (I chose from one short video)
3. There are two kinds of outputs (name, path, featMats & labels) & three numbers
		* name: 		video name
		* path:		local video path (under "UCF-101/")
		* featMats: 	total video# x feature dimension x frame#
		* labels:		total video# x 1
		* numVideo:	depend on training or testing sets (total: 13265)
		* numClass:	101
		* c_finished:	103 (no need to care...just a flag)
4. need "transforms.lua" to pre-process images
5. need to download the [Res-101](https://www.dropbox.com/s/6sjuhukma6izufi/resnet-101.t7?dl=0) model and put it into the folder 'models/'.

---

Other notes:

1. After running these codes, a new empty folder "out_frames" will be generated. You can ignore it. That's only for debugging.
2. The images in the folder "images" are only for debugging. They won't be used in the final experiment.

---
#### Contact
[Min-Hung Chen](https://www.linkedin.com/in/chensteven) at <cmhungsteve@gatech.edu>

Last updated: 05/04/2016
