### FlowNet

Motion information has been proved to be extremely important for activity recognition. Optical flow map provides the motions in a image format which can be easily integrated with CNN naturally.

One of the best Optical flow algorithm is [FlowNet](http://arxiv.org/abs/1504.06852), which utilizes appropriate CNNs to solve the optical flow estimation problem as a supervised learning task.

---
### Installation

##### FlowNet
Download FlowNet from the link [here](http://lmb.informatik.uni-freiburg.de/resources/software.php).
Follow the instructions provided from the authors for compiling the FlowNet with Caffe library.

We provide the [python scripts](https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN/tree/master/FlowNet/flownet-release/models/flownet/scripts) to process the videos using FlowNet. Because of how the FlowNet is being set up, the FlowNet will take saved image files as input. As you can see directly from [read_video.py](https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN/blob/master/FlowNet/read_video.py) and [opticalFlow_dataset.py](https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN/blob/master/FlowNet/opticalFlow_dataset.py), we save the frames into files and ask FlowNet to run which will load the images back into the caffe model. Then, again because of the last of FloeNet, the output will be saved into **.flo** file. Note that there are certainly ways to avoid these processes, but our intention is to simply use FlowNet as tool without going into too many details and modifications.

##### OpenCV-Python (Ubuntu)
We use the OpenCV library with Python to read and process the frames for each video. More information about how to use OpenCv with Python, please check the [OpenCV-Python tutorial](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html).

###### Option 1: Install OpenCV 3.0 for Python 3.4+
Follow the step-by-step install instructions from this [blog](http://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/).

Note: If you follow this, you'll install OpenCV in a virtual environment, which is recommended by the blog author.

To activate the virtual environment (You can change the environment name. Here we use 'cv' ):
```
$ workon cv
```
To leave the virtual environment:
```
$ deactivate
```

###### Option 2: Install OpenCV 2.4+ for Python 2.7+
1. follow the instructions from the [official Ubuntu document](https://help.ubuntu.com/community/OpenCV).
2. After installing OpenCV, you have to configure OpenCV. You can follow the instructions from the [blog](http://www.samontab.com/web/2014/06/installing-opencv-2-4-9-in-ubuntu-14-04-lts/).

---
### Usage
The [read_video.py](https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN/blob/master/FlowNet/read_video.py) code can read one video and generate the corresponding optical flow video using an OpenCV built-in function. Simply run the following command:
```
$ python read_video.py
```
On the other hand, the [opticalFlow_dataset.py](https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN/blob/master/FlowNet/opticalFlow_dataset.py) can read through the whole UCF-101 dataset and generate optical flow maps for each video. You will need to specify where is the UCF-101 located and the code will handle the rest. 
```
$ python opticalFlow_dataset.py
```

---
TODO:
- [x] Compile FlowNet with Caffe
- [x] install OpenCV 3+ on Python 3.4+
- [x] Write Python code with OpenCV for single video processing
- [x] Write the Python script with OpenCV to process the whole video dataset

---
#### Contact:

[Chih-Yao Ma](http://shallowdown.wix.com/chih-yao-ma) at <cyma@gatech.edu>

[Min-Hung Chen](https://www.linkedin.com/in/chensteven) at <cmhungsteve@gatech.edu>
