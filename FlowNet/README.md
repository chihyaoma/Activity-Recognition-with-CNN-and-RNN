### FlowNet

Motion information has been proved to be extremely important for activity recognition. Optical flow map provides the motions in a image format which can be easily integrated with CNN naturally.

One of the best Optical flow algorithm is [FlowNet](http://arxiv.org/abs/1504.06852), which utilizes appropriate CNNs to solve the optical flow estimation problem as a supervised learning task.

---
### Installation

##### FlowNet
Download FlowNet from the link [here](http://lmb.informatik.uni-freiburg.de/resources/software.php).
Follow the instructions provided from the authors for compiling the FlowNet with Caffe library.

##### OpenCV-Python  
We use the OpenCV library with Python to read and process the frames for each video. More information about how to use OpenCv with Python, please check the [OpenCV-Python tutorial](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html).

###### Install OpenCV 2.4+ for Python 2.7+
1. download _opencv.sh_ from this folder
2. In terminal, type:
```
$ chmod +x opencv.sh
$ ./opencv.sh
```
3. Now you have to configure OpenCV. First, open the opencv.conf file with the following code:
```
$ sudo gedit /etc/ld.so.conf.d/opencv.conf
```
Add the following line at the end of the file (it may be empty, but it's OK.) and then save it:
```
/usr/local/lib
```
Run the following code to configure the library:
```
$ sudo ldconfig
```
4. Now you have to open another file:
```
sudo gedit /etc/bash.bashrc
```
Add these two lines at the end of the file and save it:
```
PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
export PKG_CONFIG_PATH
```
Finally, **restart the computer or logout and then login again**

---
### Usage
Currently this code can read one video and generate the corresponding optical flow video using an OpenCV built-in function. Simply run the following command:
```
$ python read_video.py
```

---
TODO:
- [x] Compile FlowNet with Caffe
- [x] Write Python code with OpenCV for simple video processing
- [ ] install OpenCV 3+ on Python 3.4+
- [ ] Write the script to process the whole video dataset
- [ ] Write Python code with OpenCV for FlowNet processing


#### Contact: [Chih-Yao Ma](http://shallowdown.wix.com/chih-yao-ma) at <cyma@gatech.edu>
