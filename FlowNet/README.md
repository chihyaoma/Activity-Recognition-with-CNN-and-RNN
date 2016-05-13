### FlowNet

Motion information has been proved to be extremely important for activity recognition. Optical flow map provides the motions in a image format which can be easily integrated with CNN naturally.

One of the best Optical flow algorithm is [FlowNet](http://arxiv.org/abs/1504.06852), which utilizes appropriate CNNs to solve the optical flow estimation problem as a supervised learning task.

---
### Installation

###### FlowNet
Download FlowNet from the link [here](http://lmb.informatik.uni-freiburg.de/resources/software.php).
Follow the instructions provided from the authors for compiling the FlowNet with Caffe library.

###### OpenCV-Python  
We use the OpenCV library with Python to read and process the frames for each video. More information about how to use and install OpenCv with Python, please check the [OpenCV-Python tutorial](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html).

---
TODO:
- [x] Compile FlowNet with Caffe
- [ ] Write Python code with OpenCV for FlowNet processing


#### Contact: [Chih-Yao Ma](http://shallowdown.wix.com/chih-yao-ma) at <cyma@gatech.edu>
