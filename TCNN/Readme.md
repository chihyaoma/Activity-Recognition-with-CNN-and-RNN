# Temporal Convolutional Neural Network (TCNN)
This folder contains codes to train TCNN using the beforehand generated feature vectors of UCF101 for video classification.

## Usage & Performance:
### Command: qlua run.lua -b 10 -p float -r 15e-5

Notes:

1. Now I am using "model_Res" (2-layer architecture).
2. best performance: in the folder "results_Res/results_Res_3/"
	* [16,32,256] + [3,11] + [2,2]
	* 53 epochs
	* training: 100%
	* testing: 77.47%

---
## File List & Implementation details:
All the codes are implemented or modified by Min-Hung Chen.

### lua files:
1. run.lua:
similar as the one we used for HW3, but set the maximum epoch as 100.

3. data.lua:
similar as above but experimented on the UCF-101 split 1 with ResNet-101 features. No shuffling in order to generate the labels for demo easily.

4. train.lua:
similar as the one we used for HW3, but change "data_augmentation". We randomly selected the starting point for cropping. (We used 48 of 50 frames for each video.)
We added two more optimization methods: adam and rmsprop.

5. test.lua:
similar as the one we used for HW3, but we store the labels everytime we improve the testing
accuracy. We store the best testing accuracy as well.

9. model_Res.lua:
2-layer architecture with 1D kernels for ResNet-101 features. ==> current best results for T-CNN

11. nameList.lua:
use to generate the labels for the demo video.
	* command: th nameList_Hao.lua > labels_demo.txt

---
### text files:
1. stats.txt:
some statistics of these three models.
2. labels_demo.txt:
labels for demo

---
#### Contact
[Min-Hung Chen](https://www.linkedin.com/in/chensteven) at <cmhungsteve@gatech.edu>

Last updated: 05/04/2016
