# Baseline and Feature Generation
This folder contains codes to generate feature vectors and the baseline performance. The generated features are used for our network for temporal design: TS-LSTM and Temporal-ConvNet.

Notes:
1. Load most of the videos in UCF-101 (use CUDA for default).
2. Use the model **ResNet-101** and then fine-tune for UCF101 and HMDB51.
3. Generate the prediction results and extract the feature before the full-connected layer.
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
We used a pre-trained CNN model [Res-101](http://torch7.s3-website-us-east-1.amazonaws.com/data/resnet-101.t7) and fine-tuned for UCF101 and HMDB51 to generate the features.

We provided the fine-tuned models:

|                 | UCF101          | HMDB51      |
|:-------------:|:-------------:|:---------:|
| RGB      | [sp1](https://www.dropbox.com/s/g9elu4oo7s47pag/model_best.t7?dl=0) [sp2](https://www.dropbox.com/s/c8vsa15ldtj3la2/model_best.t7?dl=0) [sp3](https://www.dropbox.com/s/s3keajiby54hzzr/model_best.t7?dl=0) | [sp1](https://www.dropbox.com/s/ccl5r021kxe8ifl/model_best.t7?dl=0) [sp2](https://www.dropbox.com/s/w45h31s0o8967g4/model_best.t7?dl=0) [sp3](https://www.dropbox.com/s/doilqlex7yuwwyx/model_best.t7?dl=0) |
| TV-L1       | [sp1](https://www.dropbox.com/s/oi5gdzgpw20pk5x/model_best.t7?dl=0) [sp2](https://www.dropbox.com/s/7nr2k2542cmpwas/model_best.t7?dl=0) [sp3](https://www.dropbox.com/s/flafxau5elvd2nk/model_best.t7?dl=0)      |  [sp1](https://www.dropbox.com/s/coleft75y2z0deq/model_best.t7?dl=0) [sp2](https://www.dropbox.com/s/sb6xdvygisu0bjj/model_best.t7?dl=0) [sp3](https://www.dropbox.com/s/uzp33k0iu1sbgzb/model_best.t7?dl=0)  |


You may need to change the directory paths of the model and dataset (*dirModel* and *dirDatabase*) in the code (around line 116-144).

### Libraries
There are some libraries and you need to install before running my codes.

#### Reading Videos:
We used the [Torch Video Decoder Library](https://github.com/e-lab/torch-toolbox/tree/master/Video-decoder) to read videos. 

You may need to install the ffmpeg library:
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
$ th run_pred-feat_twoStreams.lua
```

### Some parameters we don't recommend to change:
numStream = 2 (because we use the two-stream network)

dimFeat = 2048 (you can change it if fine-tuning from the models other than ResNet)

### Outputs
* Features:

In each output *data_feat_xxxxxx.t7* file (e.g.data_feat_train_RGB_centerCrop_25f_sp1.t7), there are three main variables inside:

    1. name:        video name
    2. featMats:    total video# x feature dimension x frame#
    3. labels:      total video# x 1

* Predictions:

    * *data_pred_xxxxxx.t7*: all the information about the prediction
    * *acc_xxxxxx.txt*: video prediction for each class and the whole dataset


---
#### Contact
[Min-Hung Chen](https://www.linkedin.com/in/chensteven) at <cmhungsteve@gatech.edu>

Last updated: 03/27/2017
