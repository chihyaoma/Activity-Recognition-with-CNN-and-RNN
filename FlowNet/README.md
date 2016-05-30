# FlowNet

Motion information has been proved to be extremely important for activity recognition. Optical flow map provides the motions in a image format which can be easily integrated with CNN naturally.

One of the best Optical flow algorithm is [FlowNet](http://arxiv.org/abs/1504.06852), which utilizes appropriate CNNs to solve the optical flow estimation problem as a supervised learning task.

note: the main codes are in the path: *flownet-release/models/flownet/*

---
## Installation / Requirement
#### FlowNet
Download FlowNet from the link [here](http://lmb.informatik.uni-freiburg.de/resources/software.php).
Follow the instructions provided from the authors for compiling the FlowNet with Caffe library.

We provide the [python scripts](https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN/tree/master/FlowNet/flownet-release/models/flownet/scripts) to process the videos using FlowNet. Because of how the FlowNet is being set up, the FlowNet will take saved image files as input. As you can see directly from [flowNet_video_M_C++.py](https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN/blob/master/FlowNet/flownet-release/models/flownet/flowNet_video_M_C%2B%2B.py) and [flowNet_dataset.py](https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN/blob/master/FlowNet/flownet-release/models/flownet/flowNet_dataset.py), we save the frames into files and ask FlowNet to run which will load the images back into the caffe model. Then, again because of the last of FloeNet, the output will be saved into **.flo** file. Note that there are certainly ways to avoid these processes, but our intention is to simply use FlowNet as tool without going into too many details and modifications.

#### OpenCV-Python (Ubuntu)
We use the OpenCV library with Python to read and process the frames for each video. More information about how to use OpenCV with Python, please check the [OpenCV-Python tutorial](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html).

###### Option 1: Install OpenCV 3.0 for Python
Choose one of the following blogs and follow the step-by-step install instructions:
* [for Python 3.4+](http://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/)
* [for Python 2.7+](http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/)

(Note: It's not necessary to use virtual environment as mentioned in the blogs.)

###### Option 2: Install OpenCV 2.4+ for Python 2.7+
1. follow the instructions from the [official Ubuntu document](https://help.ubuntu.com/community/OpenCV).
2. After installing OpenCV, you have to configure OpenCV. You can follow the instructions from the [blog](http://www.samontab.com/web/2014/06/installing-opencv-2-4-9-in-ubuntu-14-04-lts/).

#### Python with C++
You need to compile the C++ function in the folder *colorflow_Python_C++*. Please see the readme file inside for more details. After compilation, you will get *ColorFlow.so*. Modify you code according to the path of *ColorFlow.so* and add this line in your code.
```
import ColorFlow
```
And then you can use the C++ function in the module **ColorFlow**
```
ColorFlow.flow2color(outTempName, fileName)
```

---
## Usage
The [flowNet_video_M_C++.py](https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN/blob/master/FlowNet/flownet-release/models/flownet/flowNet_video_M_C%2B%2B.py) code can read one video and generate the corresponding optical flow video using FlowNet. The flow maps will be shown in the standard Middebury color style. Simply run the following command:
```
$ python flowNet_video_M_C++.py
```
On the other hand, the [flowNet_dataset.py](https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN/blob/master/FlowNet/flownet-release/models/flownet/flowNet_dataset.py) can read through the whole UCF-101 dataset and generate optical flow maps for each video. You will need to specify where is the UCF-101 located and the code will handle the rest.
```
$ python flowNet_dataset.py
```

---
## Notes
#### Visualization
The original flow file can't be directly seen. Therefore, we adopted the standard Middlebury color encoding method to visualize the flow map. This method is used in the code *flowNet_video_M_C++.py*.

In addition, We have also tried another two visualization methods by normalizing the flow maps. One obvious way is to normalize the flow map individually. The other way is that, the normalization is being done within the *flowNet_dataset.py*. This normalization is based on the maximum value for each of the video. The purpose of doing this is so that flow map can have universal scale for the movement of the objects in the videos.

#### Other files
* *flowNet_video_M_Python.py*: implement the Middlebury color coding in Python (twice more slower)
* *flowNet_video_Nf.py*: normalization frame-by-frame
* *flowNet_video_Nv.py*: normalization through the whole video

#### Computation time
Test video: *v_Archery_g01_c06.avi*

| *flowNet_video_Nv.py* | *flowNet_video_M_Python.py* | *flowNet_video_M_C++.py* |
|:-------------:|:-------------:|:-----:|
| 34.48 s | 64.40 s | **28.19 s** |

---
##### TODO:
- [x] Compile FlowNet with Caffe
- [x] install OpenCV 3+ on Python 3.4+
- [x] Write Python code with OpenCV for single video processing
- [x] Write the Python script with OpenCV to process the whole video dataset
- [x] load the original Middlebury color coding C++ function to Python
- [ ] The script for the whole dataset with the Middlebury color style

---
### Contact:

[Chih-Yao Ma](http://shallowdown.wix.com/chih-yao-ma) at <cyma@gatech.edu>

[Min-Hung Chen](https://www.linkedin.com/in/chensteven) at <cmhungsteve@gatech.edu>

Last updated: 05/30/2016
