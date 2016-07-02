### RNN
We use the [RNN library](https://github.com/Element-Research/rnn) provided by Element-Research. This *main.lua* takes computed feature vectors from each of the frames, and learns a generic model to represent the videos aross all training video frames.

Using command line arguments or modify the code in *main.lua*, you can essentially modify the Learning rate, batch size, CUDA options, LSTM or GRU, dropout layers, and the dimension of hidden layers.

---
### Note
This is rather a note for me to remember what experiments have been done and what needs to be done in order to moving forward.

---
TODO:
- [x] Experiment with different number of frames for training
- [x] Experiment with different batch size and learning rate
- [x] Train with different optimizers, like sgd, adam, adamax, rmsprop
- [x] Comparison between GRU and LSTM, but they should pretty much perform the same
- [ ] Have all the feature vectors from all frames for testing, not just 50 frames per video
- [x] During testing, if you have at least two frames, input for the network should be feature vector from 1 to t-1
- [x] Feedforward from each time step and average across all frames over a video to make prediction
- [ ] Instead of averaging across all frames per video, can we have weightings?


Note:

Optimizer: tried SGD and Adam. It gives faster and so far the best convergence. it requires relatively smaller learning rate compared with SGD. I am using **LearningRate = 5e-4** for Adam.

Hidden layers: using hidden number all the way to 128 (1024, 768, 512, 256, 128) seems to be overfitting the data. With the feature vector dimension to be 1024, the number of hidden layers (1024, 512, 256) can achieve best accuracy.

#### Contact: [Chih-Yao Ma](http://shallowdown.wix.com/chih-yao-ma) at <cyma@gatech.edu> or [[LinkedIn]](https://www.linkedin.com/in/chih-yao-ma-9b5b3063)
